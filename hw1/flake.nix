{
  description = "Build environment for box2d-py with pip install";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        buildInputs = with pkgs; [
          # Core build tools
          gcc
          cmake
          pkg-config

          # Box2D dependencies
          swig

          # System libraries that might be needed
          zlib
          libffi
          openssl
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          inherit buildInputs;

          shellHook = ''
            echo "Box2D-py build environment activated"
            echo "SWIG: $(swig -version)"

            # Set up environment variables for building
            export PYTHONPATH="$PWD:$PYTHONPATH"
            export PIP_DISABLE_PIP_VERSION_CHECK=1
          '';
        };
      }
    );
}
