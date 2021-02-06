#Getting/Storing subreddit sentiment
import asyncpraw
import asyncio

class RedditRunner:
    def __init__(self, client_id, client_secret, user_agent, subreddit):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit = subreddit
        
    async def initialize_api(self):
        self.reddit = asyncpraw.Reddit(
            client_id = self.client_id,
            client_secret = self.client_secret,
            user_agent = self.user_agent 
        )
        print('Initializing Reddit API...')
        return self
    
    async def set_subreddit(self):
        self.subreddit = await self.reddit.subreddit(self.subreddit)
        print('Setting Subreddit')
        return self

    def get_self(self):
        return self.reddit



    


