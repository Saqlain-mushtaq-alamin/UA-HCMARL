from concurrent.futures import ProcessPoolExecutor

def square(x):
    return x * x

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(square, [1, 2, 3, 4]))
    print("MP worked:", res)

