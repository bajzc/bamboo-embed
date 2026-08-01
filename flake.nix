{
  description = "guji-rag — 中文古籍 RAG 检索系统";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      # x86_64-darwin (Intel Mac) is intentionally not in this list: the pinned
      # nixos-unstable has dropped support for it upstream (Nixpkgs 26.11+, Apple
      # Silicon only). On an Intel Mac, override the input to a still-supported
      # branch instead, e.g.:
      #   nix develop --override-input nixpkgs github:NixOS/nixpkgs/nixpkgs-25.05-darwin
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f (import nixpkgs { system = s; }));
    in {
      devShells = forAll (pkgs:
        let
          isDarwin = pkgs.stdenv.isDarwin;

          # Embeddings: `pkgs.ollama-rocm` is a Linux-only ROCm (AMD HIP) build and
          # isn't buildable on darwin at all, so it can't be used unconditionally in
          # a shell meant to evaluate on both. On Linux this dev box has an AMD GPU,
          # hence ROCm; on macOS plain `pkgs.ollama` already ships with Metal/Accelerate
          # GPU acceleration built in — no package-level flag needed. If you're on
          # Linux with an NVIDIA or no GPU, swap `ollama-rocm` below for `ollama-cuda`
          # or plain `ollama`.
          ollamaPkg = if isDarwin then pkgs.ollama else pkgs.ollama-rocm;

          # Local chat LLM: llama-server needs an explicit GPU backend flag on Linux
          # (`vulkanSupport = true`, RADV auto-detects the card, no ROCm-style gfx
          # override needed). On darwin, llama-cpp's `metalSupport` defaults to true
          # already (see nixpkgs pkgs/by-name/ll/llama-cpp/package.nix), so the plain
          # package is Metal-accelerated with no override required.
          llamaCppPkg =
            if isDarwin then pkgs.llama-cpp
            else pkgs.llama-cpp.override { vulkanSupport = true; };
        in {
        default = pkgs.mkShell {
          packages = [
            pkgs.python311
            pkgs.uv
            pkgs.git
            ollamaPkg
            llamaCppPkg
          ];

          # uv must use the nix-provided interpreter, never download its own.
          env = {
            UV_PYTHON = "${pkgs.python311}/bin/python3.11";
            UV_PYTHON_DOWNLOADS = "never";
          };

          # Linux-only: NixOS has no global /lib, so manylinux wheels (opencc, later
          # lancedb) need libstdc++ / zlib reachable via LD_LIBRARY_PATH to load their
          # .so; and the AMD GPU needs HSA_OVERRIDE_GFX_VERSION since this card
          # (gfx1032) isn't in ROCm's official support list. macOS has neither
          # problem: manylinux wheels don't apply, and Metal needs no gfx masquerade.
          shellHook = if isDarwin then "" else ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            # pretend gfx1032 is gfx1030, and use the discrete card (device 0)
            # rather than the 7900X iGPU.
            export HSA_OVERRIDE_GFX_VERSION=10.3.0
            export ROCR_VISIBLE_DEVICES=0
          '';
        };
      });
    };
}
