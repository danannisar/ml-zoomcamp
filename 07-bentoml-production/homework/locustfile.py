# import numpy as np
from locust import task
from locust import between
from locust import HttpUser


# Sample data to send
sample = [[6.4,3.5,4.5,1.2]]

# Inherit HttpUser object from locust
class MLZoomUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000, in case if all requests failed then load client with:
            locust -H http://localhost:3000 -f locustfile.py

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    # create mathod with task decorator to send request
    @task
    def classify(self):
        self.client.post("/classify", json=sample) # post request in json format with the endpoint 'classify'

    wait_time = between(0.01, 2) # set random wait time between 0.01-2 secs
