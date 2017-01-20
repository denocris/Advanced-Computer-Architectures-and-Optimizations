# CUDA GPU

Synchronous: enqueue work and wait for completion

Asynchronous: enqueue work and return immediately (Kernel lauches)

A stream is a queue of device work:
—The host places work in the queue and continues on immediately
—Device schedules work from streams when resources are free

cuda stream syncronize

cuda events

overlap