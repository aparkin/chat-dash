import pytest
from weaviate_manager.query.manager import QueryManager
from weaviate_manager.database.client import get_client
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture
def qm():
    """Fixture to provide a QueryManager instance."""
    with get_client() as client:
        yield QueryManager(client)

def test_article_retrieval(qm):
    """Test basic article retrieval functionality."""
    print('\n=== Test 1: Article Retrieval ===')
    try:
        # Get first article ID from database
        result = qm.client.collections.get("Article").query.fetch_objects(limit=1).objects
        if result:
            article_id = result[0].uuid
            print(f'Testing with article ID: {article_id}')
            
            # Test without references
            article = qm._get_article_by_id(article_id, include_references=False)
            print('Basic article retrieved with keys:', article.keys())
            print('Properties:', list(article['properties'].keys()))
            
            # Test with references
            article = qm._get_article_by_id(article_id, include_references=True)
            print('\nArticle with references:')
            print('- Authors:', len(article.get('authors', [])))
            print('- References:', len(article.get('references', [])))
            print('- Entities:', len(article.get('entities', [])))
        else:
            print('No articles found in database')
    except Exception as e:
        print(f'Error: {str(e)}')

def test_search_functionality(qm):
    """Test search functionality with different types."""
    print('\n=== Test 2: Search Functionality ===')
    try:
        # Test query
        query = "protein folding mechanisms"
        
        # Test semantic search
        print('\nTesting semantic search:')
        result = qm.search_articles(query, search_type="semantic", limit=3)
        print(f'Found {result.total_results} results')
        if result.items:
            print('First result score:', result.items[0]['metadata'].get('certainty'))
        
        # Test keyword search
        print('\nTesting keyword search:')
        result = qm.search_articles(query, search_type="keyword", limit=3)
        print(f'Found {result.total_results} results')
        if result.items:
            print('First result score:', result.items[0]['metadata'].get('score'))
        
        # Test hybrid search
        print('\nTesting hybrid search:')
        result = qm.search_articles(query, search_type="hybrid", limit=3)
        print(f'Found {result.total_results} results')
        if result.items:
            print('First result scores:')
            print('- Score:', result.items[0]['metadata'].get('score'))
            print('- Certainty:', result.items[0]['metadata'].get('certainty'))
    except Exception as e:
        print(f'Error: {str(e)}')

def test_reference_handling(qm):
    """Test reference handling functionality."""
    print('\n=== Test 3: Reference Handling ===')
    try:
        # Get first article ID
        result = qm.client.collections.get("Article").query.fetch_objects(limit=1).objects
        if result:
            article_id = result[0].uuid
            print(f'Testing with article ID: {article_id}')
            
            # Test outgoing references
            refs = qm._get_article_references(article_id, limit=5)
            print(f'\nFound {len(refs)} outgoing references')
            if refs:
                print('First reference properties:', list(refs[0]['properties'].keys()))
            
            # Test incoming citations
            cites = qm._get_article_citations(article_id, limit=5)
            print(f'\nFound {len(cites)} incoming citations')
            if cites:
                print('First citation properties:', list(cites[0]['properties'].keys()))
            
            # Test reference exploration
            print('\nTesting reference exploration:')
            result = qm.explore_references(article_id, direction="outgoing", depth=2, limit=3)
            print(f'Found {result.total_results} references in network')
    except Exception as e:
        print(f'Error: {str(e)}')

def test_invalid_article(qm):
    """Test error handling with invalid article ID."""
    print('\n=== Test 4: Invalid Article ===')
    try:
        invalid_id = '00000000-0000-0000-0000-000000000000'
        article = qm._get_article_by_id(invalid_id)
        print('Unexpected success!')
    except Exception as e:
        print(f'Expected error received: {str(e)}')

if __name__ == '__main__':
    # Use client manager as context manager
    with get_client() as client:
        # Initialize QueryManager with client
        qm = QueryManager(client)
        
        # Run tests
        test_article_retrieval(qm)
        test_search_functionality(qm)
        test_reference_handling(qm)
        test_invalid_article(qm) 