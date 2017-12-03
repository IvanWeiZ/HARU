import boto3
import json

import table
# Get the service resource.
dynamodb = boto3.resource('dynamodb')
from boto3.dynamodb.conditions import Key, Attr


# Create the DynamoDB table.
def create_table():
    table = dynamodb.create_table(
        TableName='users',
        KeySchema=[
            {
                'AttributeName': 'username',
                'KeyType': 'HASH'
            },
            {
                'AttributeName': 'last_name',
                'KeyType': 'RANGE'
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'username',
                'AttributeType': 'S'
            },
            {
                'AttributeName': 'last_name',
                'AttributeType': 'S'
            },

        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )

    # Wait until the table exists.
    table.meta.client.get_waiter('table_exists').wait(TableName='users')

    # Print out some data about the table.
    print(table.item_count)


# Instantiate a table resource object without actually
# creating a DynamoDB table. Note that the attributes of this table
# are lazy-loaded: a request is not made nor are the attribute
# values populated until the attributes
# on the table resource are accessed or its load() method is called.
#table = dynamodb.Table('users')

# Print out some data about the table.
# This will cause a request to be made to DynamoDB and its attribute
# values will be set based on the response.
# print(table.creation_date_time)
# table.put_item(
#    Item={
#         'username': 'janedoe',
#         'first_name': 'Jane',
#         'last_name': 'Doe',
#         'age': 25,
#         'account_type': 'standard_user',
#     }
# )
#
# response = table.get_item(
#     Key={
#         'username': 'janedoe',
#         'last_name': 'Doe'
#     }
# )
# item = response['Item']
# print(item)

#
# response = table.query(
# )
# items = response['Items']
# print(items)
#
# response = table.scan(
#     FilterExpression=Attr('age').lt(27)
# )
# items = response['Items']
# print(items[0].values())


locationTable=table.scan_table("Location","Devicename","1")
#print(items)
# print(locationTable)
# print(len(locationTable))
# print(locationTable["Items"])
print(len(locationTable["Items"]))
with open('../outputfileNov30', 'w') as fout:
    #print(locationTable["Items"])
    json.dump(locationTable["Items"], fout)



