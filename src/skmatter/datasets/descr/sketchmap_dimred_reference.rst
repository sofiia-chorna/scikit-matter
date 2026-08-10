.. _sketchmap-dimred-reference:

sketch-map reference embedding
##############################

Reference two-dimensional embedding produced by the C++ ``dimred`` of the original
sketch-map implementation (`sketchmap.org <https://sketchmap.org>`_) for the first 64
handwritten digits of :func:`sklearn.datasets.load_digits`, each 64-dimensional image.
It is used to validate :class:`skmatter.decomposition.SketchMap` against the reference
implementation. The output of ``dimred`` is deterministic, so shipping the embedding
makes the comparison reproducible without building the C++ program.

Function Call
-------------

.. function:: skmatter.datasets.load_sketchmap_dimred_reference

Data Set Characteristics
------------------------

:Number of Instances: 64

:Number of Features: 2

:Reported stress: 0.0129757

:Sigmoid parameters: ``sigma=30``, ``a_high=4``, ``b_high=2``, ``a_low=2``, ``b_low=2``

The embedding was computed with the three ``dimred`` calls below: an iterative metric
MDS, the sigmoid fit initialized from it, and a final grid global optimization. This is
the pipeline that ``utils/sketch-map.sh`` drives, prompting for the settings and
assembling the calls itself. The ``-fun-hd`` and ``-fun-ld`` flags each take the
switching distance and the two steepness exponents of the high- and low-dimensional
sigmoid, so ``-fun-hd 30,4,2 -fun-ld 30,2,2`` is the parameter set listed above:

.. code-block:: bash

    dimred -D 64 -d 2 -center -preopt 100
    dimred -D 64 -d 2 -center -preopt 100 \
           -fun-hd 30,4,2 -fun-ld 30,2,2 -init tmp
    dimred -D 64 -d 2 -center -preopt 100 -grid <gw>,21,201 \
           -fun-hd 30,4,2 -fun-ld 30,2,2 -init tmp -gopt 10

References
----------

[1] Michele Ceriotti, Gareth A. Tribello, and Michele Parrinello. "Simplifying
    the representation of complex free-energy landscapes using sketch-map." Proceedings
    of the National Academy of Sciences 108.32 (2011): 13023-13028.

Reference Code
--------------

[2] https://github.com/lab-cosmo/sketchmap
