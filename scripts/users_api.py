import csv
import os
import requests
import time
import threading


API_URL = "https://zara-boost-hackathon.nuwe.io/users"
PATH = "../../data/raw"
DATASET = "users.csv"
VARIABLES = ["user_id", "country", "R", "F", "M"]
# To check if the request was answered correctly
isOK = lambda request: request.status_code == 200

def _save_as_csv(data: list):
    """
    Once 10k or less records are registered, they are stored in 
    the csv file corresponding to the USERS dataset.

    Parameters
    ----------
    data: list
        Matrix of users features       
    """
    with open(PATH + '/'+ DATASET, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for record in data:
            writer.writerow(record)

def _format_entry(user_data: dict) -> list:
    """
    Format the user's info. in csv format

    Parameters
    ----------
    user_data: dict
        JSON record of a user
    
    Return
    ------
    list
        List of user's features 
    """
    entry = [user_data[VARIABLES[0]]]
    entry += [user_data["values"][v][0] for v in VARIABLES[1:]] 
    return entry


def _start_thread(task, params: tuple) -> threading.Thread:
    """
    Thread that retrieve a data of a given batch of users_id
    
    Parameters
    -----------
    users_id
        Range object of users id
    """
    thread = threading.Thread(target=task, args=(params))
    thread.start()
    return thread

def _get_user_info(u_id: int) -> list:
    """
    Get the information of a particular user in
    json format.

    Parameters
    ----------
    u_id: int
        User's id
    
    Return
    ------
    list
        User's attributes
    """
    user_request = lambda u_id: requests.get(API_URL + f"/{u_id}")
    retrieved = False
    while not retrieved:
        try:
            record = user_request(u_id)
            retrieved = True
        except:
            time.sleep(2)
    if isOK(record):
        record = _format_entry(record.json())
        return record

def _load_data(user_ids):
    """
    Get the information of the users to build the dataset.

    Parameters
    ----------
    api_url: str
        URL to get the the information of the user
    user_ids: list
        List with user ids to use them as parameter in the API's URL
    """
    data = []
    omitted = []
    # To manage the number of registered records
    rows = 0
    for u_id in user_ids:
        print(u_id)
        record = _get_user_info(u_id)
        if record is not None:
            data.append(record)
            rows += 1
            if rows == pow(10, 4): # Once 10k records have been recorded
                _start_thread(_save_as_csv, (data.copy(),))
                data.clear()
                time.sleep(2)
                rows = 0
        else:
            omitted.append(u_id)
    if len(data) != 0: # If there are still users to register.
        _save_as_csv(data)
    if len(omitted) > 0:
        print(f"\nOmitted User IDs:\n{omitted}")

def batching(users_id: list):
    """
    Import the USERS data by applying batching and
    multithreading.
    
    Parameters
    ----------
    users_id: list
        List of all user ids
    """
    N = len(users_id)
    start = 0
    step = N // 4 # Split in 4 batches
    batches = []
    for _ in range(4):
        if len(batches) == 3:
            batch = users_id[start:]
        else:
            end = start + step
            batch = users_id[start:end]
        start += step
        batches.append(_start_thread(_load_data, (batch,)))
        time.sleep(5)
    # PCT of data stored
    pct = 25
    for batch in batches:
        batch.join()
        print(f"{pct}% of data stored")
        pct += 25
    print("\n USERS dataset have been created correctly")

def isCompleted() -> bool | None: 
    """
    It checks if it has been registered the information of 
    all the user_ids. It's useful when this script fail for
    some reason, since it's not necessary to load all the data
    again. 
    """
    N = requests.get(API_URL)
    if isOK(N):
        total_ids = set(range(1, len(N.json()) + 1))
        with open(PATH + '/' + DATASET, 'r') as csvfile:
            reader = csv.reader(csvfile)
            registered = {int(row[0]) for row in reader}
        rest_ids = list(total_ids.difference(registered))
        if len(rest_ids) != 0:
            batching(rest_ids)
        else:
            print("The USERS dataset is completed.")

if __name__ == "__main__":
    if DATASET not in os.listdir(PATH):
        users_id = requests.get(API_URL)
        if isOK(users_id):
            users_id = range(1, len(users_id.json()) + 1) 
            batching(users_id)
        else:
            print("STATUS CODE:{users_id.status_code}")
    else:
        isCompleted()
