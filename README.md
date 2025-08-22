# sat-val-framework
Framework for implementing satellite validation. Contains class definitions for handling raw data, collocation events, quality checks and data homogenisation.

A basic schematic for the framework is given in the Figure below:

![Satellite validation schematic](./satellite-validation-framework.svg)

Raw data is handled by a collocation scheme to identify collocation events, within the maximally allowed collocation parametrisations.
The collocation events are stored on disk, and can subsequently be used to load all collocation events allowed with a given parametrisation.
The raw data classes also handle quality checking of the data.
Homogenised data classes are implemented, and raw data implememnt methods to be transformed to match a specific homogenised data format.
Homogenised data are then used in any subsequent analysis.
