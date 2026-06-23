from importlib import import_module

REQUIRED_MODULES = [
    "dvc",
    "fastapi",
    "mlflow",
    "numpy",
    "pandas",
    "sklearn",
    "torch",
]


def main() -> None:
    missing = []
    for module in REQUIRED_MODULES:
        try:
            import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(f"Dependencias ausentes: {', '.join(missing)}")
    print("Ambiente validado com sucesso.")


if __name__ == "__main__":
    main()
