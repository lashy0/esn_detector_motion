from collections import deque
from typing import Any, List


class FixedBuffer:
    """
    Fixed-size buffer.

    Args:
        size (int): The maximum size of the buffer.
    """
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = deque(maxlen=size)
    
    def add(self, data: Any) -> None:
        """
        Add a new value to the buffer.
        If the buffer is full, the oldest value will be removed.

        Args:
            data (Any): The data to add to the buffer.
        """
        self.buffer.append(data)
    
    def get_buffer(self) -> List:
        """
        Retrieve the current state of the buffer.

        Returns:
            List: A list containing the current items in the buffer.
        """
        return list(self.buffer)
    
    def get_value(self, index: int) -> Any:
        """
        Get the value at the specified index in the buffer.

        Args:
            index (int): The index of the value to retrieve.

        Returns:
            Any: The value at the specified index.
        """
        if index < 0:
            index = len(self.buffer) + index
        
        if index < 0 or index >= len(self.buffer):
            raise IndexError("Index out of range")
        return self.buffer[index]
    
    def get_current_size(self) -> int:
        """
        Get the current number of elements in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all values from buffer, resetting it to an empty state."""
        self.buffer.clear()
    
    def is_full(self) -> bool:
        """
        Check whether the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return len(self.buffer) == self.size
    
    def __repr__(self):
        return f"FixedBuffer({self.get_buffer()})"
