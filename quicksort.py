def quick_sort(array):
    def sort(array, low, high):
        if low >= high:
            return
        i = low
        k = array[i]
        p = low
        for j in range(low+1, high+1):
            if array[j] < k:
                array[i], array[j] = array[j], array[i]
                i += 1
        array[array.index(k)], array[i] = array[i], array[array.index(k)]
        sort(array, low, i - 1)
        sort(array, i + 1, high)
    sort(array, 0, len(array) - 1)


if __name__ == "__main__":
    array = [2, 7, 8, 9, 10, 1]
    quick_sort(array)
    print(array)