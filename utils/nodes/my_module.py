def my_function(param1, param2, optional_param=None):
    """
    This is a short, concise summary of my_function.

    A more detailed explanation of what the function does.
    It can span multiple lines and describe the overall purpose.
    You might include examples of usage here.

    Parameters
    ----------
    param1 : int
        Description of the first parameter. It should be an integer
        representing some quantity.
    param2 : str
        Description of the second parameter. This is a string that
        could be a name or a path.
    optional_param : list of float, optional
        An optional parameter, defaults to None. This should be a list
        of floating-point numbers.

    Returns
    -------
    bool
        True if the operation was successful, False otherwise.
        A more detailed explanation of the return value can go here.

    Raises
    ------
    ValueError
        If `param1` is negative or `param2` is an empty string.
    TypeError
        If `param1` is not an integer.

    See Also
    --------
    another_function : Relevant function for related operations.
    some_class.method : A related method of a class.

    Notes
    -----
    This section can contain any additional information about the function,
    such as algorithms used, limitations, or best practices.
    It's good for conveying context not directly related to parameters or returns.

    Examples
    --------
    >>> my_function(10, "hello")
    True
    >>> my_function(5, "world", optional_param=[1.0, 2.5])
    True
    >>> my_function(-1, "test")
    Traceback (most recent call last):
        ...
    ValueError: param1 cannot be negative.
    """

    return True
