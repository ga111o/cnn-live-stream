import time
import asyncio
from recognize import recognize_image
from icecream import ic

async def process_batch(start, end):
    for i in range(start, end):
        result = await recognize_image(f'./test_images/3.jpg')
        ic(i, result)

async def main():
    start_time = time.time()
    
    # 100개의 작업을 6개의 배치로 나눔
    batch_size = 300 // 6
    tasks = []
    
    for i in range(6):
        start = i * batch_size + 1
        end = start + batch_size if i < 5 else 101
        tasks.append(process_batch(start, end))
    
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    _time = end_time - start_time
    ic(_time)

if __name__ == "__main__":
    asyncio.run(main())