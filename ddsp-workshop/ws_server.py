import websockets
import asyncio
import threading 

def worker(loop, port, callback):
    asyncio.set_event_loop(loop)
    # https://pypi.org/project/websockets/
    async def echo(websocket, path):
        async for message in websocket:
            callback(message)
            #print("wsserve.py revceived message", message)
            # send it back!
            await websocket.send(message)
    print("wsserve.py running websocket server")

    loop.run_until_complete(
        websockets.serve(echo, 'localhost', port)
    )
    loop.run_forever()
    
def run_websocket(callback, port=8765):
    """
    starts a websocket server in a separate thread
    on the specified port (or 8765) by default
    """
    # https://stackoverflow.com/questions/48725890/runtimeerror-there-is-no-current-event-loop-in-thread-thread-1-multithreadi
    loop = asyncio.new_event_loop()
    p = threading.Thread(target=worker, args=(loop,port,callback))
    p.start()


