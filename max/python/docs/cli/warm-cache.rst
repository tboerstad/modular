:title: max warm-cache


Preloads and compiles one or more models to optimize initialization time by:

- Pre-compiling models before deployment
- Warming up the Hugging Face cache
- Taking advantage of shared kernels when precompiling multiple models

This command is useful to run before serving a model.

For example, to precompile a single model:

.. code-block:: bash

    max warm-cache \
      --model google/gemma-3-12b-it

To precompile multiple models in a single invocation, use ``--additional-model``.
Models that share the same architecture (e.g., two Llama variants) benefit from
shared kernel compilation -- the compilation engine caches compiled kernel objects
so redundant work is avoided:

.. code-block:: bash

    max warm-cache \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --additional-model meta-llama/Llama-3.1-70B-Instruct \
      --additional-model meta-llama/Llama-3.2-3B-Instruct

.. raw:: markdown

    :::note

    The Modular Executable Format (MEF) is platform independent, but
    the serialized cache (MEF files) produced during compilation is
    platform-dependent. This is because:

    - Platform-dependent optimizations happen during compilation.
    - Fallback operations assume a particular runtime environment.

    Weight transformations and hashing during MEF caching can impact performance.
    While efforts to improve this through weight externalization are ongoing,
    compiled MEF files remain platform-specific and are not generally portable.

    :::

.. click:: max.entrypoints.pipelines:cli_warm_cache
  :prog: max warm-cache
  :hide-description:
