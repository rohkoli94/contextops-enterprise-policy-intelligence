from app.providers.storage.azure_blob import AzureBlobStorageProvider


storage_provider = AzureBlobStorageProvider() #custom implementation which we have created

print(
    f"Connected to container: "
    f"{storage_provider.container_client.container_name}"
)

# run below command to run this test
# uv run python -m scripts.test_blob_storage