# FFCx-backends

> [!WARNING]
> This project is under heavy development and is no where close to stable.

*FFCx-backends* extends the FEniCS Form Compiler ([FFCx](https://github.com/fenics/ffcx)) by providing multiple other language backends as plugins.

Usage through FFCx's CLI is straight forward by passing any of the supported language modules with the `--language` argument.

```console
    ffcx --language ffcx-backends.[lang] form.py
```

This supports any [UFL](https://github.com/fenics/ufl) script compatible with classic `C` backend of FFCx.

## Supported backends

| Language | Status             |
| -------- | -------------------|
| C++      | 🛠️ experimental    |
| CUDA     | ⏳ in development  |
| ?        | 💡 to be suggested |

## Contributing

> [!NOTE]  
> In preparation. We are happy to include any other backend - open an issue for furhter discussion!
