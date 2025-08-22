"""Author: Andrew Martin
Creation date: 22/08/2025

Class definitions that can be used without modification in the validation process.
"""


class CollocationEventList(UserList):
    """Handles a list of collocation events, and saving them to and loading them from files"""
    def __init__(self, data):
        super().__init__(self, data)
        for i,entry in enumerate(data):
            assert isinstance(entry, CollocationEvent), f"{data[i]=}, of type {type(entry)}, is not an instance of CollocationEvent"

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
        super().__init__(self, data)

    @classmethod
    def from_collocation_event_list_and_parameters(cls, event_list: CollocationEventList, parameters: CollocationParameters) -> Self:
        assert isinstance(event_list, CollocationEventList), f"{type(event_list)} must be an instance of {CollocationEventList}"
        assert isinstance(parameters, CollocationParameters), f"{type(parameters)} must be an instance of {CollocationParameters}"
        new_data = ( # TODO: fix a way that an instance of CollocatedRawDataList knows what subclass of CollocatedRawData (with defined R1 and R2) that it should be loading
            CollocatedHomogenisedData
        )
        

class CollocatedHomogenisedDataList(UserList):
    def __init__(self, data):
        super().__init__(self, data)
        for i, value in enumerate(data):
            assert isinstance(i, HomogenisedData), f"{data[i]=} should be subclass of HomogenisedData, is of type {type(data)}"
        
    @property
    def events(self):
        return self.data



