import multiprocessing as mp
import multiprocessing.connection as mpc
import select
import time

from src.stopwatch import Stopwatch


def _worker(_id: int, pipe, parent_pipe, interval_sec: float):
    parent_pipe.close()
    pipe.send(_id)
    while True:
        cont = pipe.recv()

        if not cont:
            break

        time.sleep(interval_sec)
        pipe.send(_id)
    pipe.close()


def main():
    stopwatch = Stopwatch()

    parent_pipes: list[mp.Pipe] = []
    processes: list[mp.Process] = []

    for _id in [3, 5, 7]:
        parent_pipe, child_pipe = mp.Pipe()
        process = mp.Process(
            target=_worker,
            name=f'Worker-{_id}',
            args=(
                _id,
                child_pipe,
                parent_pipe,
                _id
            )
        )

        parent_pipes.append(parent_pipe)
        processes.append(process)

        process.start()
        child_pipe.close()

    try:
        stopwatch.reset()
        while parent_pipes:
            for r in mpc.wait(parent_pipes):
                _id = r.recv()
                print(f'{stopwatch.time_passed():>5}: {_id}')
                if stopwatch.time_passed() < 10:
                    r.send(True)
                else:
                    r.send(False)
                    parent_pipes.remove(r)
    finally:

        for parent_pipe, process in zip(parent_pipes, processes):
            parent_pipe.close()
            process.close()


if __name__ == '__main__':
    main()
