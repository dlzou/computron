import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any, Hashable


@dataclass
class SubmitEntry:
    uid: Hashable
    data: Any

@dataclass
class LoadEntry:
    uid: Hashable
    load: bool
    blocking: bool


class Engine:
    def __init__(self):
        self.submit_queue = deque() # queue for model requests
        self.submit_cv = asyncio.Condition()
        self.can_submit = False
        self.work_queue = asyncio.Queue()
        self.completion_map = {}
        self.completion_event = {}
    
    async def handle_entry(self, entry):
        if isinstance(entry, SubmitEntry):
            self.completion_event[entry.uid] = asyncio.Event()
            self.submit_queue.append(entry)
            await self.completion_event[entry.uid].wait()

            del self.completion_event[entry.uid]
            return self.completion_map.pop(entry.uid)

        elif isinstance(entry, LoadEntry):
            # Acquire submit_cv
            async with self.submit_cv:
                # await self.submit_cv.wait_for(lambda: self.can_submit)
                if self.can_submit != entry.load:
                    if entry.blocking:
                        self.completion_event[entry.uid] = asyncio.Event()
                        await self.work_queue.put(entry)
                        await self.completion_event[entry.uid].wait()
                        del self.completion_event[entry.uid]
                        output = self.completion_map.pop(entry.uid)
                    else:
                        await self.work_queue.put(entry)
                        output = entry
                    self.can_submit = entry.load
                    if self.can_submit:
                        self.submit_cv.notify()
            return output

    async def submit_loop(self):
        while True:
            if len(self.submit_queue) > 0:
                # Acquire submit_cv
                async with self.submit_cv:
                    await self.submit_cv.wait_for(lambda: self.can_submit)
                    entry = self.submit_queue.popleft()
                    await self.work_queue.put(entry)
                    self.submit_cv.notify()
            else:
                await asyncio.sleep(0.1)
    
    async def worker(self):
        while True:
            entry = await self.work_queue.get()
            if isinstance(entry, SubmitEntry):
                print(f"worker running {entry.data}")
            elif isinstance(entry, LoadEntry):
                print(f"worker running load={entry.load}")

            await asyncio.sleep(0.5)
            if isinstance(entry, SubmitEntry) or entry.blocking:
                self.completion_map[entry.uid] = entry
                self.completion_event[entry.uid].set()
            await asyncio.sleep(0.1)


async def client(engine):
    uid = 0
    while True:
        output = await engine.handle_entry(SubmitEntry(uid, f"entry {uid}"))
        print(f"client got {output}")
        await asyncio.sleep(0.5)
        uid += 2


async def controller(engine):
    uid = 1
    while True:
        output = await engine.handle_entry(LoadEntry(uid, load=True, blocking=False))
        print(f"controller got {output}")
        await asyncio.sleep(3)
        output = await engine.handle_entry(LoadEntry(uid, load=False, blocking=True))
        print(output)
        await asyncio.sleep(3)
        uid += 2


engine = Engine()
loop = asyncio.new_event_loop()
loop.create_task(client(engine))
loop.create_task(controller(engine))
loop.create_task(engine.worker())
loop.create_task(engine.submit_loop())
loop.run_forever()