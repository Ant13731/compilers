def select(f, array):
    """Loop & branch"""
    result = []
    for element in array:
        if f(element):
            result.append(element)
    return result
