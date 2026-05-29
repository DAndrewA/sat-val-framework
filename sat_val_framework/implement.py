"""Creation date: 22/08/2025
Author: Andrew Martin

Script containing class definitions for the sat-val-framework package
"""

from __future__ import annotations

from typing import Self, Type, ClassVar, Optional
from dataclasses import dataclass, asdict
from collections import UserDict

import os
import pickle



@dataclass(kw_only=True, frozen=True)
class RawDataEvent:
    """Class that should be implemented with all fields sufficient to load raw data with no subsetting or further information.
    """
    # identifies the RawData type the class is associated with
    RDT: ClassVar[Type[RawData]] = None
    pass



@dataclass(kw_only=True, frozen=True)
class RawDataSubsetter:
    """Class that handles collocation subsetting based on a parametrisation"""
    # identifies the RawData type the class is associated with
    RDT: ClassVar[Type[RawData]] = None

    def subset(self, raw_data: RawData) -> Optional[RawData]:
        raise NotImplementedError(f"{type(self)} does not implement subset method")



@dataclass(kw_only=True)
class RawMetadata:
    """Class handling metadata for RawData classes"""
    loader: str | RawDataEvent
    subsetter: list[RawDataSubsetter]



class RawData:
    """Class to handle raw data from an arbitrary source.

    METHODS:
        assert_on_creation(self) -> None | AssertionError
        @classmethod from_qualified_file(cls, fpath: str) -> Self
        @classmethod from_collocation_event_and_parameters(cls, event: RawDataEvent, parameters: RawDataSubsetter) -> Self
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
    def from_qualified_file(cls, fpath: str) -> Optional[Self]:
        raise NotImplementedError(f"Type {type(self)} does not implement .from_qualfied_file(cls, fpath: str)")

    @classmethod
    def from_collocation_event_and_parameters(cls, event: RawDataEvent, parameters: CollocationParameters) -> Optional[Self]:
        raise NotImplementedError(f"Type {type(self)} does not implement .from_collocation_event_and_parameters(cls, event: RawDataEvent, parameters: CollocationParameters)")

    def perform_qc(self) -> Self:
        raise NotImplementedError(f"Type {type(self)} does not implement .perform_qc(self)")

    @property
    def n_profiles(self) -> int:
        raise NotImplementedError(f"Type {type(self)} does not implement .n_profiles property")

    def homogenise_to(self, H: Type[HomogenisedData]) -> H:
        raise NotImplementedError(f"Type {type(self)} does not implement .homngenise_to(self, H: Type[HomogenisedData])")



class HomogenisedData:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.assert_on_creation()

    def assert_on_creation(self) -> None | Exception:
        raise NotImplementedError("Type {type(self)} does not implement .assert_on_creation(self)")



class CollocationScheme:
    @staticmethod
    def get_matches_from_raw_directories(raw_directory1: str, raw_directory2: str) -> CollocationEventList:
        raise NotImplementedError(f"Type {type(self)} does not implement .get_matches_from_raw_directories(raw_directory1: str, raw_directory2: str)")



class JointParameters(UserDict):
    """Class that handles RawDataSubsetter instances per RawData type in the analysis"""
    RAW_DATA_TYPES: tuple[RawData] = tuple()

    def __init__(self, data: dict[Type[RawData], RawDataSubsetter]):
        assert set(self.RAW_DATA_TYPES) == set(data.keys()), f"data keys contain different RawData types to {self.RawDataTypes}"
        for RDT, params in data.items():
            if isinstance(params, RawDataSubsetter):
                assert params.RDT == RDT, ValueError(f"For key={RDT} in data, {params.RDT=} does not match.")
            else
                assert params is None, ValueError(f"For key={RDT} in data, params must be of type None or {RawDataSubsetter} with correctly set RDT field.")
        super().__init__(data)



class CollocationEvent(UserDict):
    def __init__(self, data: dict[Type[RawData], RawDataEvent]):
        for RDT, event in data.items():
            assert issubclass(RDT, raw_data), TypeError(f"Key={RDT} in data is not a subclass of {RawData}.")
            assert isinstance(event, RawDataEvent), TypeError(f"Data supplied for key {RDT} is of type {type(raw_data)}, should be a subclass of {RDT}.")
            assert event.RDT == RDT, ValueError(f"{event.RDT=} must match the {RDT=}")
        super().__init__(data)

    @property
    def events(self): return self.data

    def load_with_joint_parameters(self, joint_params: JointParameters) -> Optional[CollocatedRawData]:
        raw_datas = {
            RDT: RDT.from_collocation_event_and_parameters(
                event = event,
                parameters = joint_params[RDT]
            )
            for RDT, event in self.data.items()
        }
        for raw_data in raw_datas.values():
            if raw_data is None:
                return None
        return CollocatedRawData(
            data = raw_datas
        )


class CollocatedRawData(UserDict):
    def __init__(self, data: dict[Type[RawData], RawData]):
        for RDT, raw_data in data.items():
            assert issubclass(RDT, RawData), TypeError(f"Key={RDT} in data is not a subclass of {RawData}.")
            assert isinstance(raw_data, RDT), TypeError(f"Data supplied for key {RDT} is of type {type(raw_data)}, should be a subclass of {RDT}.")
        super().__init__(data)

    def subset(self, joint_parameters: JointParameters) -> Optional[Self]:
        # TODO: Raw Data Type checks on the JointParameters and Self
        subset_raw_data = {
            RDT: (
                subsetter.subset(raw_data)
                if (subsetter := joint_parameters[RDT]) is not None
                else raw_data
            )
            for RDT, raw_data in self.data.items()
        }
        for raw_data in subset_raw_data.values():
            if raw_data is None:
                return None
        return type(self)(
            data = subset_raw_data
        )

    def homogenise_to(self, H: Type[HomogenisedData]) -> CollocatedHomogenisedData:
        homogenised_datas = {
            RDT: raw_data.homogenise_to(H)
            for RDT, raw_data in self.data.items()
        }
        return CollocatedHomogenisedData(H, homogenised_datas)



class CollocatedHomogenisedData(UserDict):
    def __init__(self, H: Type[HomogenisedData], data: dict[Type[RawData], HomogenisedData]):
        assert issubclass(H, HomogenisedData), f"Cannot create CollocatedHomogenisedData holding {H} that is not a subclass of HomogenisedData"
        assert all((
            issubclass(RDT, RawData) & isinstance(homogenised_data, H)
            for RDT, homogenised_data in data.items()
        )), f"All keys must be Type[{RawData}] and all values must be {H}"
        
        super().__init__(data)
        self.H = H
