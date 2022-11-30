import boto3

import os

location = {'LocationConstraint': 'us-east-2'}


# client.create_bucket(Bucket = 'testprogaccess', CreateBucketConfiguration=location)


def uploadFolder(path, bucketname, client): #uploads a path to a bucket
    for root, dirs, files in os.walk(path):
        for file in files:
            print("Uploading: " + str(os.path.join(root, file)))
            client.upload_file(os.path.join(root, file), bucketname, file)


def deleteBucketObjects(bucketname, client): #this function deletes all files in a bucket
    #PREFIX = 'folder1/'
    response = client.list_objects_v2(Bucket=bucketname)
    print(response)
    if 'Contents' in response:
        for object in response['Contents']:
            print('Deleting', object['Key'])
            client.delete_object(Bucket=bucketname, Key=object['Key'])


if __name__ == "__main__":
    client = boto3.client('s3')
    response = client.list_buckets()

    # delete objects before uploading new ones
    deleteBucketObjects('testprogaccess', client)

    #specify path to images folder on nano
    #uploadFolder(path = '/Users/manavbhasin/Desktop/test_aws', bucketname='testprogaccess', client=client)




