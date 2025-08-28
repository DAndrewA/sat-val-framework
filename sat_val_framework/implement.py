"""Creation date: 22/08/2025
Author: Andrew Martin

Script containing class definitions for the sat-val-framework package
"""

from __future__ import annotations

from typing import Self, Type
from dataclasses import dataclass, asdict

import os
import pickle


@dataclass(kw_only=True)
class RawMetadata:
    """Class handling metadata for RawData classes"""
    loader: str | CollocationEvent
    subsetter: list[RawDataSubsetter]

@dataclass(kw_only=True, frozen=True)
class RawDataSubsetter:
    """Class that handles collocation subsetting based on a parametrisation"""
    def subset(self, raw_data: RawData) -> RawData:
        raise NotImplementedError(f"{type(self)} does not implement subset method")


class RawData:
    """Class to handle raw data from an arbitrary source.

    METHODS:
        assert_on_creation(self) -> None | AssertionError
        @classmethod from_qualified_file(cls, fpath: str) -> Self
        @classmethod from_collocation_event_and_parameters(cls, event: CollocationEvent, parameters: RawDataSubsetter) -> Self
        perform_qc(self) -> Self
        homogenise_to(self, H: Type[HomogenisedData]) -> H
    """

    def __init__(self, data, metadata: RawMetadata):
        self.data = data
        self.metadata = metadata
        self.assert_on_creation()
        self.perform_qc()

    def assert_on_creation(self) -> None | Exception:
        raise AssertionError(f"Type {type(self)} does not implement .assert_on_creation()")

    @classmethod
    def from_qualified_file(cls, fpath: str) -> Self:
        raise NotImplementedError(f"Type {type(self)} does not implement .from_qualfied_file(cls, fpath: str)")

    @classmethod
    def from_collocation_event_and_parameters(cls, event: CollocationEvent, parameters: CollocationParameters) -> Self:
        raise NotImplementedError(f"Type {type(self)} does not implement .from_collocation_event_and_parameters(cls, event: CollocationEvent, parameters: CollocationParameters)")

    def perform_qc(self) -> Self:
        raise NotImplementedError(f"Type {type(self)} does not implement .perform_qc(self)")

    def homogenise_to(self, H: Type[HomogenisedData]) -> H:
        raise NotImplementedError(f"Type {type(self)} does not implement .homngenise_to(self, H: Type[HomogenisedData])")



class HomogenisedData:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.assert_on_creation()

    def assert_on_creation(self) -> None | Exception:
        raise NotImplementedError("Type {type(self)} does not implement .assert_on_creation(self)")



class RawDataPairBaseClass: 
    """Inherit from this class and RawDataPairMetaclass if the checks in RawDataPairMetaclass.__new__ are to be bypassed."""
    pass

class RawDataPairMetaclass(type):
    """Metaclass that validates that class attributes R1 and R2 are set as subclasses of RawData upon class definition"""
    def __new__(cls, name, bases, namespace):
        # create the class initially
        new_class = super().__new__(cls, name, bases, namespace)

        # skip any validation if RawDataPairBaseClass is a direct parent
        if RawDataPairBaseClass in bases:
            return new_class

        R1 = getattr(new_class, "R1", None)
        if not isinstance(R1, type):
            raise ValueError(f"Class definition for {name} should set {R1=} as a subclass of {RawData}")
        assert issubclass(R1, RawData), f"Class definition for {name} should set {R1=} as a subclass of {RawData}"

        R2 = getattr(new_class, "R2", None)
        if not isinstance(R2, type):
            raise ValueError(f"Class definition for {name} should set {R2=} as a subclass of {RawData}")
        assert issubclass(R2, RawData), f"Class definition for {name} should set {R2=} as a subclass of {RawData}"

        return new_class



@dataclass(frozen=True, kw_only=True)
class CollocationParameters:
    def apply_collocation_subsetting(self, raw_data: RawData) -> RawData:
        raise NotImplementedError(f"Type {type(self)} does not implement .apply_collocation_subsetting(self, raw_data: RawData)")

    def calculate_collocation_criteria(self, raw_data1: RawData, raw_data2: RawData) -> tuple[RawData, RawData]:
        raise NotImplementedError(f"Type {type(self)} does not implement .calculate_collocation_criteria(self, raw_data1: RawData, raw_data2: RawData)")

    def get_collocation_event(self, raw_data1: RawData, raw_data2: RawData) -> CollocationEvent | Exception:
        raise NotImplementedError(f"Type {type(self)} does not implement .get_collocation_event(self, raw_data1: RawData, raw_data2: RawData)")



class CollocationScheme:
    @staticmethod
    def get_matches_from_raw_directories(raw_directory1: str, raw_directory2: str) -> CollocationEventList:
        raise NotImplementedError(f"Type {type(self)} does not implement .get_matches_from_raw_directories(raw_directory1: str, raw_directory2: str)")



@dataclass(frozen=True, kw_only=True)
class CollocationEvent:
    def to_dict(self) -> dict:
        return asdict()

    @classmethod
    def from_dict(cls, d: dict) -> Self | Exception:
        return cls(**d)



class CollocatedRawData(RawDataPairBaseClass, metaclass=RawDataPairMetaclass):
    R1 = None
    R2 = None

    def __init__(self, raw_data1: RawData, raw_data2: RawData):
        assert isinstance(raw_data1, self.R1), f"{type(raw_data1)=} is not {self.R1=}"
        assert isinstance(raw_data2, self.R2), f"{type(raw_data2)=} is not {self.R2=}"
        self.raw_data1 = raw_data1
        self.raw_data2 = raw_data2

    @classmethod
    def from_collocation_event_and_parameters(cls, event: CollocationEvent, parameters: CollocationParameters) -> Self:
        raw_data1 = self.R1.from_collocation_event_and_parameters(
            event = event,
            parameters = parameters
        )
        raw_data2 = self.R2.from_collocation_event_and_parameters(
            event = event,
            parameters = parameters
        )
        return cls(raw_data1=raw_data1, raw_data2=raw_data2)

    def homogenise_to(self, H: H) -> H:
        homogenised1 = self.raw_data1.homogenise_to(H)
        homogenised2 = self.raw_data2.homogenise_to(H)
        return CollocatedHomogenisedData[H](homogenised1, homogenised2)



class CollocatedHomogenisedData:
    def __init__(self, homogenised1: HomogenisedData, homogenised2: HomogenisedData):
        assert isinstance(homogenised1, HomogenisedData), f"{type(homogenised1)=} is not instance of {HomogenisedData}"
        assert isinstance(homogenised2, HomogenisedData), f"{type(homogenised2)=} is not instance of {HomogenisedData}"
        self.homogenised1 = homogenised1
        self.homogenised2 = homogenised2
