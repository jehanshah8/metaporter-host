import boto3
import os
import glob

def downloadObjects(client, bucketname, img_store_path): #downloads all files in a bucket and stores locally
    response = client.list_objects_v2(Bucket=bucketname)
    # print(response)
    if 'Contents' in response:
        for i, object in enumerate(response['Contents']):
            print('Downloading', object['Key'])
            #client.delete_object(Bucket=bucketname, Key=object['Key'])
            client.download_file(bucketname, object['Key'], img_store_path + str(i) + '.png')

def cleanFolder(path):
    files = glob.glob(path + '*')
    for file in files:
        os.remove(file)

if __name__ == "__main__":
    client = boto3.client('s3')

    img_store_path = '/Users/manavbhasin/Downloads/test_buck_download/'
    bucket_name = 'testprogaccess'
    cleanFolder(img_store_path)
    downloadObjects(client, bucket_name, img_store_path)




