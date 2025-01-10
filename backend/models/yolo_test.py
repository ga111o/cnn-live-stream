import time
import asyncio
from yolo_recognize import recognize_image, _load_models
from icecream import ic

async def process_batch(start, end):
    for i in range(start, end):
        result = await recognize_image(f'./test_images/3.jpg')
        ic(i, result)

async def main():
    # Load model once before starting concurrent operations
    _load_models()
    
    start_time = time.time()
    
    batch_size = 300 // 1
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