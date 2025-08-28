"""Author: Andrew Martin
Creation date: 22/08/2025

Class definitions that can be used without modification in the validation process.
"""

from __future__ import annotations

from .implement import (
    CollocationEvent, 
    JointParameters, 
    CollocatedRawData, 
    CollocatedHomogenisedData, 
    HomogenisedData,
)

from typing import Self
from collections import UserList

import os
import pickle



class CollocationEventList(UserList):
    """Handles a list of collocation events, and saving them to and loading them from files"""
    def __init__(self, data):
        # assert all elements in the list are CollocationEvent instances
        assert all((
            isinstance(element, CollocationEvent)
            for element in data
        )), f"All entries should be of type {CollocationEvent}"
        
        # assert all CollocationEvent instances have the same RawData types as keys
        RDT_KEYS0 = set(data[0].events.keys())
        assert all((
            set(event.events.keys()) == RDT_KEYS0
            for event in data
        )), f"All events should have the same RawData type keys, event[0].RDTs={RDT_KEYS0}" 
        super().__init__(self, data)


    def load_with_joint_parameters(self, joint_params: JointParameters) -> CollocatedRawDataList:
        #TODO: implement safety into these methods
        return CollocatedRawDataList((
            collocation_event.load_with_join_parameters(joint_params)
            for collocation_event in self.data
        ))


    def to_file(fpath: str):
        if os.path.exists(fpath):
            print(f"File already exists at {fpath=}, not saving output")
        with open(fpath, "wb") as f:
            pickle.dump(self, fpath)

    @classmethod
    def from_file(cls, fpath: str) -> Self:
        with open(fpath, "rb") as f:
            pickle_ob = pickle.load(f)
            assert isinstance(pickle_ob, cls), f"Pickle loading instance of type {type(pickle_ob)}, not of required type {cls}"
        return pickle_ob



class CollocatedRawDataList(UserList):
    def __init__(self, data):
        assert all((
            isinstance(collocated_raw_data, CollocatedRawData)
            for collocated_raw_data in data
        )), f"All elements should be [CollocatedRawData]"
        super().__init__(self, data)

    def homogenise_to(self, H: Type[HomogenisedData]) -> CollocatedHomogenisedDataList:
        return CollocatedHomogenisedDataList((
            collocated_raw_data.homogenise_to(H)
            for collocated_raw_data in self.data
        ))
        

class CollocatedHomogenisedDataList(UserList):
    def __init__(self, data):
        super().__init__(self, data)
        for i, value in enumerate(data):
            assert isinstance(i, HomogenisedData), f"{data[i]=} should be subclass of HomogenisedData, is of type {type(data)}"
        
    @property
    def events(self):
        return self.data



