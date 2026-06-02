import pytest
from . import implement

class TestReturnCaughtErrors():
    @implement.return_caught_errors(ValueError, TypeError)
    def fn(self, a: int):
        match a:
            case 1: return "yippee"
            case 2: raise ValueError("a != 1")
            case 3: raise TypeError("a is not int[==1]")
            case 4: raise IndexError("a is not an index")
            case 5:
                assert a == 1, ValueError("a == 5, not 1")
            case 6:
                assert a == 1, IndexError("a is still not an index")

    @implement.return_caught_errors(unpack_assertion_error=False)
    def fn_without_unpack(sef):
        assert 6 == 1, ValueError("6 != 1")

    @implement.return_caught_errors()
    def fn_without_decorator_arguments(self):
        raise ValueError("Decorator should have had arguments")

    def test_good_return(self):
        assert self.fn(1) == "yippee"

    def test_caught_errors(self):
        assert type(self.fn(2)) is ValueError
        assert type(self.fn(3)) is TypeError

    def test_raising_unchecked_for_errors(self):
        with pytest.raises(IndexError):
            self.fn(4)

    def test_good_unpack_assertion_error(self):
        assert type(self.fn(5)) is ValueError

    def test_bad_unpack_assertion_error(self):
        with pytest.raises(IndexError):
            self.fn(6)

    def test_no_unpacking(self):
        with pytest.raises(AssertionError):
            self.fn_without_unpack()

    def test_no_decorator_arguments(self):
        with pytest.raises(ValueError):
            self.fn_without_decorator_arguments()
