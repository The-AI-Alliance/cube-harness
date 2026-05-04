import csv
import logging

logger = logging.getLogger(__name__)


# I want to write a function, reads a csv file, and get all the contents in the third column in the order of rows
def get_conference_city_in_order(env, config):
    # read the csv file
    csv_path = config["csv_path"]
    logger.debug("get_conference_city_in_order: reading csv from %s", csv_path)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        # skip the header row
        next(reader)
        # get the third column in the order of rows
        conference_city_list = [row[2] for row in reader]
    return conference_city_list
