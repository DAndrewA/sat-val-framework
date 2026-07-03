"""Creation date: 22/08/2025
Author: Andrew Martin

Script containing class definitions for the sat-val-framework package
"""

from __future__ import annotations

from typing import Self, Type, ClassVar, Optional, Callable, Union, Any
from functools import wraps
from dataclasses import dataclass, asdict
from collections import UserDict

import os
import pickle


class InvalidSubsetError(ValueError):
    pass
type LoadingErrors = FileNotFoundError | InvalidSubsetError


def return_caught_errors(*types_list: list[Type], unpack_assertion_error: bool = True, check_assertion_error_type: bool=True):
    """A function wrapper that catches specified error types and returns them as values instead.

    INPUTS:
        *types_list: list[Type]:
            Exception subclasses that should be caught and returned. Raised exceptions that are not subclasses of any value in types_list will still be raised.

        unpack_assertion_error: bool
            If True, and AssertionError is not present in types_list, AssertionError instances will be caught and returned.

        check_assertion_error_type: bool
            If True, when unpacking an AssertionError, the inner argument's type will be checked against types_list, and the inner error will be raised if it does not match.
    """
    type f = Callable[...,Any]
    type g = Callable[..., Union[Any,*types_list]]
    def return_caught_errors_decorator[F:f, G:g](func: F) -> G:
        @wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except types_list as E:
                return E
            except AssertionError as AE:
                if unpack_assertion_error:
                    print(AE, AE.args,type(AE.args), type(AE.args[0]))
                    inner_error = AE.args[0]
                    if not check_assertion_error_type:
                        return inner_error
                    if isinstance(inner_error, tuple(types_list)):
                        return inner_error
                    raise inner_error
                raise AE
        return _wrapper
    return return_caught_errors_decorator
        


@dataclass(kw_only=True, frozen=True)
class RawDataEvent:
    """Class that should be implemented with all fields sufficient to load raw data with no subsetting or further information.
    """
    # identifies the RawData type the class is associated with
    RDT: ClassVar[Type[RawData]] 
    pass



@dataclass(kw_only=True, frozen=True)
class RawDataSubsetter:
    """Class that handles collocation subsetting based on a parametrisation"""
    # identifies the RawData type the class is associated with
    RDT: ClassVar[Type[RawData]] 

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
        raise NotImplementedError(f"Type {cls} does not implement .from_qualfied_file(cls, fpath: str)")

    @classmethod
    def from_collocation_event_and_parameters(cls, event: RawDataEvent, parameters: CollocationParameters) -> Optional[Self]:
        raise NotImplementedError(f"Type {cls} does not implement .from_collocation_event_and_parameters(cls, event: RawDataEvent, parameters: CollocationParameters)")

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
    @classmethod
    def get_matches_from_raw_directories(cls, raw_directory1: str, raw_directory2: str) -> CollocationEventList:
        raise NotImplementedError(f"Type {cls} does not implement .get_matches_from_raw_directories(raw_directory1: str, raw_directory2: str)")



class JointParameters(UserDict):
    """Class that handles RawDataSubsetter instances per RawData type in the analysis"""
    RAW_DATA_TYPES: tuple[RawData]

    def __init__(self, data: dict[Type[RawData], RawDataSubsetter]):
        for RDT in data.keys():
            if issubclass(RDT, self.RAW_DATA_TYPES):
                continue
            raise AssertionError(KeyError("data key {data_RDT} is not found in {self.RAW_DATA_TYPES=}"))
                
        for RDT, params in data.items():
            if isinstance(params, RawDataSubsetter):
                assert issubclass(RDT, params.RDT), ValueError(f"For key={RDT} in data, {params.RDT=} does not match.")
            else:
                assert params is None, ValueError(f"For key={RDT} in data, params must be of type None or {RawDataSubsetter} with correctly set RDT field.")
        super().__init__(data)



class CollocationEvent(UserDict):
    def __init__(self, data: dict[Type[RawData], RawDataEvent]):
        for RDT, event in data.items():
            assert issubclass(RDT, RawData), TypeError(f"Key={RDT} in data is not a subclass of {RawData}.")
            assert isinstance(event, RawDataEvent), TypeError(f"Data supplied for key {RDT} is of type {type(event)}, should be a subclass of {RawDataEvent}.")
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

    #TODO: determine what errors should be passed through
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
