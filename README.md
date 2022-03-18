# `simquil` simulates [Quil](https://quil-lang.github.io)

... but not very quickly.

`simquil` is
  - a hobby project to learn a bit about:
    - simulating quantum programs
    - rust 🦀

`simquil` is not
  - fast
  - better than [QVM](https://github.com/quil-lang/qvm)
  - complete

# Try it out

```shell
$ echo "H 0; CNOT 0 1; CNOT 1 2" | cargo run

|000>: 0.707107+0.000000i, 50%
|111>: 0.707107+0.000000i, 50%
```
