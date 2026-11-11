"""Retrieve library poses with local Korean sentence embeddings."""

def index_text(row):
    description = row.get('description') or {}
    tags = [tag.get('label', '') for group in row.get('groups', {}).values()
            for tag in group if tag['field'] not in ('intent', 'emotion')]
    return ' '.join([description.get('text') or '', *tags])
