"""Creation date: 22/08/2025
Author: Andrew Martin

Script containing class definitions for the sat-val-framework package
"""

from __future__ import annotations

from typing import Self, Type, ClassVar, TypeVar
from dataclasses import dataclass, asdict
from collections import UserList



class RawData:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.assert_on_creation()

    def assert_on_creation(self) -> None | Exception:
        raise AssertionError(f"Type {type(self)} does not implement .assert_on_creation()")

    @classmethod
    def from_qualified_file(cls, fpath: str) -> Self:
        raise NotImplementedError(f"Type {type(self)} does not implement .from_qualfied_file(cls, fpath: str)")

    @classmethod
    def from_collocation_event_and_parameters(cls, event: CollocationEvent, parameters: CollocationParameters) -> Self:
        raise NotImplementedError(f"Type {type(self)} does not implement .from_collocation_event_and_parameters(cls, event: CollocationEvent, parameters: CollocationParameters)")

    def homogenise_to(self, H: Type[HomogenisedData]) -> H:
        raise NotImplementedError(f"Type {type(self)} does not implement .homngenise_to(self, H: Type[HomogenisedData])")



class HomogenisedData:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.assert_on_creation()

    def assert_on_creation(self) -> None | Exception:
        raise NotImplementedError("Type {type(self)} does not implement .assert_on_creation(self)")



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


H = TypeVar('H', bound=HomogenisedData)
class CollocatedRawData[R1: RawData, R2: RawData]:
    def __init__(self, raw_data1: R1, raw_data2: R2):
        assert isinstance(raw_data1, R1), f"{type(raw_data1)=} is not {R1}"
        assert isinstance(raw_data2, R2), f"{type(raw_data2)=} is not {R2}"
        self.raw_data1 = raw_data1
        self.raw_data2 = raw_data2

    @classmethod
    def from_collocation_event_and_parameters(cls, event: CollocationEvent, parameters: CollocationParameters) -> Self:
        raw_data1 = R1.from_collocation_event_and_parameters(
            event = event,
            parameters = parameters
        )
        raw_data2 = R2.from_collocation_event_and_parameters(
            event = event,
            parameters = parameters
        )
        return cls(raw_data1=raw_data1, raw_data2=raw_data2)

    def homogenise_to(self, H: H) -> H:
        homogenised1 = self.raw_data1.homogenise_to(H)
        homogenised2 = self.raw_data2.homogenise_to(H)
        return CollocatedHomogenisedData[H](homogenised1, homogenised2)



class CollocatedHomogenisedData[H: HomogenisedData]:
    def __init__(self, homogenised1: H, homogenised2: H):
        assert isinstance(homogenised1, H), f"{type(homogenised1)=} is not {H}"
        assert isinstance(homogenised2, H), f"{type(homogenised2)=} is not {H}"
        self.homogenised1 = homogenised1



class CollocationEventList(UserList): pass

class CollocatedRawData(UserList): pass

class CollocatedHomogenisedData(UserList): pass



