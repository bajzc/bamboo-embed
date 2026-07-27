{
  description = "guji-rag — 中文古籍 RAG 检索系统 (Phase 1 corpus tooling)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f (import nixpkgs { system = s; }));
    in {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          # ollama-rocm = AMD HIP build (pkgs.ollama is CPU-only). The dev box has a
          # Radeon RX 6750 GRE (gfx1032), which ROCm doesn't officially support, so
          # HSA_OVERRIDE_GFX_VERSION masquerades it as gfx1030 (see shellHook).
          packages = [ pkgs.python311 pkgs.uv pkgs.git pkgs.ollama-rocm ];

          # uv must use the nix-provided interpreter, never download its own.
          env = {
            UV_PYTHON = "${pkgs.python311}/bin/python3.11";
            UV_PYTHON_DOWNLOADS = "never";
          };

          # NixOS has no global /lib; manylinux wheels (opencc, later lancedb)
          # need libstdc++ / zlib reachable via LD_LIBRARY_PATH to load their .so.
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            # AMD GPU for Ollama: pretend gfx1032 is gfx1030, and use the discrete
            # card (device 0) rather than the 7900X iGPU.
            export HSA_OVERRIDE_GFX_VERSION=10.3.0
            export ROCR_VISIBLE_DEVICES=0
          '';
        };
      });
    };
}
