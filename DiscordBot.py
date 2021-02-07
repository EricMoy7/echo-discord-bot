import os
import discord
# from dotenv import load_dotenv
from Reddit import RedditRunner
import asyncio
import time

#TODO: Create a getter function for discord channels

class DiscordBot:
    def __init__(self):

        # load_dotenv()
        self.TOKEN = os.getenv('DISCORD_TOKEN')

        return

    async def reddit_to_discord(self, client):
        await client.wait_until_ready()
        #Move keys to env file
        runner = RedditRunner("2hKYgXAVthCQvA","A1K8JN_20eOXOw3SpZhGNCEoZyd3Vg","airzm", "robinhoodpennystocks+pennystocks+stocks")
        #robinhoodpennystocks+pennystocks
        await runner.initialize_api()
        await runner.set_subreddit()

        #Replace this with getter function
        channel = client.get_channel(776287381547515943)

        while not client.is_closed():
            try:
                async for submission in runner.subreddit.stream.submissions(skip_existing=True):
                    print(f'Time: {time.ctime(time.time())}')
                    if submission:
                        embedVar = discord.Embed(title=f'{submission.title}', description=f'Written by: {submission.author}')
                        embedVar.add_field(name="Link", value=submission.url, inline=False)
                        embedVar.add_field(name="Content", value=submission.selftext, inline=False)
                        await channel.send(embed=embedVar)
                        print('Sending Message...')
                    

            except Exception as e:
                print(e)
                await asyncio.sleep(60)
                continue
    
    def initialize_bot(self):
        client = discord.Client()

        @client.event
        async def on_ready():
            print('Bot started!')

        @client.event
        async def on_message(message: str) -> None:
            if message.content.startswith('!hello'):
                embedVar = discord.Embed(title="Something About Stocks", description="Even More stuff about stocks")
                embedVar.add_field(name="Field1", value="hi", inline=False)
                embedVar.add_field(name="Field2", value="hi2", inline=False)
                await message.channel.send(embed=embedVar)
        
        client.loop.create_task(self.reddit_to_discord(client))
        client.run(self.TOKEN)
            