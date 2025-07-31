{
  description = "A Nix-flake-based Go 1.22 development environment";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1";
    sops-nix.url = "github:Mic92/sops-nix";
    sops-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs:
    let
      goVersion = 24; # Change this to update the whole stack

      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: inputs.nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.self.overlays.default ];
        };
      });
    in
    {
      overlays.default = final: prev: {
        go = final."go_1_${toString goVersion}";
      };

      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            # go (version is specified by overlay)
            go

            # goimports, godoc, etc.
            gotools

            # https://github.com/golangci/golangci-lint
            golangci-lint

            # Task runner
            go-task

            # Hot reload for Go development
            air

            # For secrets management
            sops
          ];

          # Environment variables for sops
          shellHook = ''
            export SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt
          '';
        };
      });

      packages = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.buildGoModule {
          pname = "claude-code-open";
          version = "0.3.0"; # You might want to make this dynamic, e.g., from git describe
          src = ./.; # Source is the current directory
          mainPackage = "./cmd";

          # IMPORTANT: You need to calculate this hash.
          # Run `nix build .#claude-code-open` once, it will fail and tell you the correct hash.
          # Or, run `nix-prefetch-go --mod-file go.mod` in the project directory.
          vendorHash = "sha256-rBDPd2E2zNUiXImEYkudUWiGNJcKFWQ+vkK8WoHJB1I=";

          # Go build flags
          ldflags = [ "-s" "-w" ];

          installPhase = ''
            mkdir -p $out/bin
            cp $GOPATH/bin/claude-code-open $out/bin/cco
          '';
        };
      });

      apps = forEachSupportedSystem ({ pkgs }: {
        cco = {
          type = "app";
          program = "${inputs.self.packages.${pkgs.system}.default}/bin/cco";
        };
      });
    };
}