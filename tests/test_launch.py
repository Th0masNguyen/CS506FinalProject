import pytest
from app import app

@pytest.fixture
def client():
    """Set up the Flask test client."""
    with app.test_client() as client:
        yield client

def test_app_launch(client):
    """Test if the Flask app launches and serves the home route."""
    response = client.get("/")
    assert response.status_code == 200
