{
  description = "Flake for sliding_features-rs";

  inputs = {
    nixpks.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    rust-overlay,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rust = (
          pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-src"
              "rust-analyzer"
              "clippy"
            ];
            targets = ["x86_64-unknown-linux-gnu"];
          }
        );
        cargo_upgrades = pkgs.rustPlatform.buildRustPackage {
          name = "cargo-upgrades";
          src = builtins.fetchGit {
            url = "https://gitlab.com/kornelski/cargo-upgrades";
            rev = "95e1d282dd165c69f0eb4dc66a09db5265734f54";
          };
          useFetchCargoVendor = true;
          cargoHash = "sha256-yEUfWe4/kSvBPx3xneff45+K3Gix2QXDjUesm+psUxI=";
          doCheck = false; # Tests fail at the current revision.
          meta = {
            description = "Check for outdated dependencies in a cargo workspace";
            homepage = "https://gitlab.com/kornelski/cargo-upgrades";
          };
        };
        buildInputs = with pkgs; [
          rust
          openssl
          protobuf
          clang
          pkg-config
          fontconfig
          cmake
        ];
        tools = with pkgs; [
          # Use nightly formatter, but otherwise stable channel
          (lib.hiPrio rust-bin.nightly."2026-02-01".rustfmt)
          taplo
          cargo-semver-checks
          cargo_upgrades
        ];
        nix_tools = with pkgs; [
          alejandra # Nix code formatter
          deadnix # Nix dead code checker.
          statix # Nix static code checker.
        ];
      in
        with pkgs; {
          devShells.default = mkShell {
            buildInputs = buildInputs ++ tools ++ nix_tools;
          };
        }
    );
}
