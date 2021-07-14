
import asyncio
import websockets

# https://pypi.org/project/websockets/
async def echo(websocket, path):
    async for message in websocket:
        print("ws revceived message", message)
        await websocket.send(message)

asyncio.get_event_loop().run_until_complete(
    websockets.serve(echo, 'localhost', 8765))
asyncio.get_event_loop().run_forever()

