# FFCx-backends

[![DOI](https://zenodo.org/badge/1128445944.svg)](https://doi.org/10.5281/zenodo.23034915)

---

> [!WARNING]
> This project is under heavy development.

_FFCx-backends_ extends the FEniCS Form Compiler ([FFCx](https://github.com/fenics/ffcx)) by providing other language backends as plugins.

Usage through FFCx's CLI is by passing any of the supported language modules with the `--language` argument.

```console
    ffcx --language ffcx_backends.[lang] form.py
```

This supports any [UFL](https://github.com/fenics/ufl) script compatible with classic `C` backend of FFCx.

## Supported backends

| Language | Status             |
| -------- | ------------------ |
| C++      | 🛠️ experimental    |
| CUDA     | 🛠️ experimental    |
| ?        | 💡 to be suggested |

## Contributing

> [!NOTE]  
> In preparation. We are happy to include any other backend - open an issue for further discussion!
