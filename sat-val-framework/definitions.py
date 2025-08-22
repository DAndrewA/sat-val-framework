"""Creation date: 22/08/2025
Author: Andrew Martin

Script containing class definitions for the sat-val-framework package
"""

from __future__ import annotations

from typing import Self, Type
from dataclasses import dataclass, asdict
from collections import UserList



class RawData:
    def __init__(self, data, metadata):
        raise NotImplementedError(f"Type {type(self)} does not implement __init__.")

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

    def assert_on_creation(self):
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

class CollocatedRawData: pass

class CollocatedHomogenisedData: pass

class CollocationEventList(UserList): pass

class CollocatedRawData(UserList): pass

class CollocatedHomogenisedData(UserList): pass



