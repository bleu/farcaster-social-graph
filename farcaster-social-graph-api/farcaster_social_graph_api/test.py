import asyncio
import concurrent.futures
import psutil
import os

def cpu_bound_work():
    # Use a more modest nice value
    os.nice(5)
        
    result = 0
    for i in range(10**7):
        result += i
    return result

async def limited_cpu_task():
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(pool, cpu_bound_work)
    return result

async def normal_task():
    print("Running normal priority task")
    await asyncio.sleep(1)
    return "done"

async def main():
    async with asyncio.TaskGroup() as tg:
        cpu_task = tg.create_task(limited_cpu_task())
        normal_tasks = [tg.create_task(normal_task()) for _ in range(5)]
    print("All tasks completed")

if __name__ == '__main__':
    # This is required for multiprocessing to work correctly
    asyncio.run(main())