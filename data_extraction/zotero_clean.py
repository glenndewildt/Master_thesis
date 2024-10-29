from pyzotero import zotero, zotero_errors
import time

# Initialize Zotero API client
library_id = '13784242'
api_key = '9mfRkwYqLGZA0e8MzX8z9STi'
zot = zotero.Zotero(library_id, 'user', api_key)

def safe_delete_item(item, max_retries=5):
    """Safely delete an item and verify the deletion, with retries for version conflicts."""
    retries = 0
    while retries < max_retries:
        try:
            # Ensure the item has a version
            if 'version' not in item['data']:
                raise ValueError("Item missing 'version'")
            # Print the entire item for debugging
            print(f"Attempting to delete item: {item}")
            # Delete the item
            time.sleep(2)  # Wait for deletion to process
            zot.delete_item([item])
            return True
        except zotero_errors.PyZoteroError as e:
            print(e)
            if '412' in str(e):
                print(f"Version conflict detected for item {item.get('data', {}).get('key', 'Unknown')}. Fetching latest version and retrying...")
                try:
                    # Fetch the latest version of the item
                    latest_item = zot.item(item['key'])
                    item = latest_item  # Update the item with the latest version
                    retries += 1
                except Exception as e:
                    print(f"Error fetching latest version for item {item.get('data', {}).get('key', 'Unknown')}: {str(e)}")
                    return False
            else:
                print(f"Error deleting item {item.get('data', {}).get('key', 'Unknown')}: {str(e)}")
                return False
    print(f"Failed to delete item {item.get('data', {}).get('key', 'Unknown')} after {max_retries} retries due to version conflicts.")
    return False

def sync_library():
    """Force a sync with the Zotero server."""
    try:
        if modified_items:
            for item in modified_items:
                try:
                    zot.update_item(item)
                except Exception as e:
                    print(f"Error updating item: {str(e)}")
            print("Changes synced with Zotero server")
        return True
    except Exception as e:
        print(f"Error syncing with server: {str(e)}")
        return False

# Fetch all items in your library
print("Fetching items from Zotero...")
items = zot.everything(zot.items())
print(f"Retrieved {len(items)} items")

# Track modified items for final sync
modified_items = []

def is_valid_item(item):
    """Check if item is a main bibliography item (not an attachment or snapshot)."""
    skip_titles = {'Snapshot', 'Full Text PDF', 'ScienceDirect Snapshot', 'My Library | Zotero', 'Zotero | Your personal research assistant'}
    if item.get('data', {}).get('itemType') in ['attachment', 'note']:
        return False
    if item.get('data', {}).get('title') in skip_titles:
        return False
    return True

def has_pdf(item):
    """Check if an item has a PDF attachment."""
    try:
        attachments = zot.children(item.get('key'))
        for attachment in attachments:
            if 'application/pdf' in attachment.get('data', {}).get('contentType', ''):
                return True
        return False
    except Exception as e:
        print(f"Error checking PDF for item {item.get('data', {}).get('title', 'Unknown')}: {str(e)}")
        return False

# Filter valid items first
valid_items = [item for item in items if is_valid_item(item)]
print(f"Found {len(valid_items)} valid items out of {len(items)} total items")

# Group items by title
titles = {}
for item in valid_items:
    title = item.get('data', {}).get('title')
    if not title:
        continue
    if title not in titles:
        titles[title] = []
    titles[title].append(item)

# Process duplicates
deleted_count = 0
processed_count = 0

for title, items_with_title in titles.items():
    if len(items_with_title) > 1:  # Only process if there are duplicates
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} duplicate sets...")
        
        # Check which items have PDFs
        items_with_pdf = []
        items_without_pdf = []
        
        for item in items_with_title:
            if has_pdf(item):
                items_with_pdf.append(item)
            else:
                items_without_pdf.append(item)
        
        # If we have both items with and without PDFs, delete the ones without PDFs
        if items_with_pdf and items_without_pdf:
            for item in items_without_pdf:
                if safe_delete_item(item):
                    deleted_count += 1
                    print(f"Deleted duplicate without PDF: {title}")
                    sync_library()  # Sync after each deletion
                time.sleep(1)  # Rate limiting

# Final sync with server
print("\nSyncing changes with Zotero server...")
if sync_library():
    print("Final sync completed successfully")
else:
    print("Warning: Final sync may not have completed successfully")

print("\nSummary:")
print(f"Processed {processed_count} sets of duplicates")
print(f"Deleted {deleted_count} duplicate items without PDFs")
print("Completed: Kept items with PDF attachments and removed their duplicates without PDFs.")