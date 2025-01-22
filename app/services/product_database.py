import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import settings


@dataclass
class Product:
    id: str
    name: str
    price: float
    description: str
    features: str
    match_explanation: str
    metadata: Dict


class ProductDatabase:
    def __init__(self):
        # Initialize ChromaDB client
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = chromadb.Client()

        # Use OpenAI's embedding function
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name="text-embedding-ada-002"
        )

        # Get or create the product collection
        self.collection = self.client.get_or_create_collection(
            name="products", embedding_function=self.embedding_fn, metadata={"hnsw:space": "cosine"}
        )

    async def get_all_products(self) -> List[Product]:
        """Retrieve all products from the database"""
        # Get the total count of items in the collection
        collection_data = self.collection.get()

        products = []
        if collection_data["ids"]:  # Check if there are any products
            for idx, doc_id in enumerate(collection_data["ids"]):
                metadata = collection_data["metadatas"][idx]
                doc = collection_data["documents"][idx]
                products.append(
                    Product(
                        id=doc_id,
                        name=metadata["name"],
                        price=float(metadata["price"]),
                        description=doc,
                        features=metadata.get("features", ""),
                        match_explanation="",  # No match explanation for get_all
                        metadata=metadata,
                    )
                )

        return products

    def _build_search_query(
        self,
        product_type: Optional[str] = None,
        preferences: List[str] = [],
        constraints: List[str] = [],
        features: List[str] = [],
        brand: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        query_parts = []
        if product_type:
            query_parts.append(f"looking for {product_type}")
        if preferences:
            query_parts.append(f"with preferences: {', '.join(preferences)}")
        if constraints:
            query_parts.append(f"considering constraints: {', '.join(constraints)}")
        if brand:
            query_parts.append(f"Brand : {brand}")
        if category:
            query_parts.append(f"Category : {category}")
        return " ".join(query_parts)

    def _filter_by_price(self, min_price: Optional[float] = None, max_price: Optional[float] = None) -> Dict:
        """Create price filter conditions for ChromaDB query"""
        conditions = []
        
        # Add minimum price condition
        if min_price is not None and min_price > 0:
            conditions.append({"price": {"$gt": min_price}})
            
        # Add maximum price condition
        if max_price is not None and max_price > 0:
            conditions.append({"price": {"$lt": max_price}})
            
        # If we have both conditions, combine them with $and
        if len(conditions) > 1:
            return {"$and": conditions}
        # If we have one condition, return it directly
        elif conditions:
            return conditions[0]
        # If no conditions, return empty dict
        return {}

    async def search_products(
        self,
        query: Optional[str] = None,
        product_type: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        preferences: List[str] = [],
        constraints: List[str] = [],
        brand: Optional[str] = None,
        category: Optional[str] = None,
        features: List[str] = [],
        n_results: int = 5,
    ) -> List[Product]:
        # Build search query from criteria
        search_query = self._build_search_query(
            product_type, preferences, constraints, features=features, brand=brand, category=category
        )

        if not search_query and query:
            search_query = query
        elif not search_query:
            search_query = "all products"

        # Build where clause with proper operator structure
        where_conditions = []

        # Handle category matching with variations
        if category:
            category_lower = category.lower()
            if "laptop" in category_lower:
                categories = ["laptop", "laptops", "gaming laptop", "gaming laptops"]
                where_conditions.append({"$or": [{"category": cat} for cat in categories]})
            else:
                where_conditions.append({"category": category_lower})

        # Handle brand matching
        if brand:
            brand_upper = brand.upper()
            where_conditions.append(
                {"$or": [{"brand": brand_upper}, {"brand": brand.lower()}, {"brand": brand.title()}]}
            )

        # Add price filters if specified
        price_filters = self._filter_by_price(min_price, max_price)
        if price_filters:
            where_conditions.append(price_filters)

        # Enhance search query with features for better vector matching
        if features:
            feature_query = " ".join(features)
            search_query = f"{search_query} with features: {feature_query}"

        # Construct the final where clause
        where_clause = {}
        if len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}
        elif len(where_conditions) == 1:
            where_clause = where_conditions[0]

        # Debug prints
        print(f"\nDebug - Search Query: {search_query}")
        print(f"Debug - Where Conditions: {where_conditions}")
        print(f"Debug - Final Where Clause: {where_clause}")

        # Search in ChromaDB with increased n_results for better filtering
        query_params = {
            "query_texts": [search_query],
            "n_results": n_results * 2,  # Get more results for post-filtering
        }

        if where_clause:
            query_params["where"] = where_clause

        try:
            results = self.collection.query(**query_params)

            # Debug print results
            print("\nDebug - Query Results:")
            print(f"Documents found: {len(results['documents'][0]) if results['documents'] else 0}")
            if results["documents"] and len(results["documents"][0]) > 0:
                print("First result metadata:", results["metadatas"][0][0])

            # Convert results to Product objects with post-filtering for features
            products = []
            if results["documents"] and len(results["documents"]) > 0:
                for idx, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][idx]

                    products.append(
                        Product(
                            id=results["ids"][0][idx],
                            name=metadata["name"],
                            price=float(metadata["price"]),
                            description=doc,
                            features=metadata.get("features", ""),
                            match_explanation=self._generate_match_explanation(
                                query or search_query, doc, results["distances"][0][idx]
                            ),
                            metadata=metadata,
                        )
                    )

            # Limit to requested number of results after post-filtering
            products = products[:n_results]

            print(f"\nDebug - Products found: {len(products)}")
            for product in products:
                print(f"- {product.name} (${product.price})")

            return products
        except Exception as e:
            print(f"Debug - ChromaDB query error: {str(e)}")
            raise

    async def get_recommendations(
        self,
        product_type: Optional[str] = None,
        preferences: List[str] = [],
        constraints: List[str] = [],
        n_results: int = 3,
    ) -> List[Product]:
        # For recommendations, we'll use the same search but with different ranking
        return await self.search_products(
            product_type=product_type, preferences=preferences, constraints=constraints, n_results=n_results
        )

    def _generate_match_explanation(self, query: str, doc: str, distance: float) -> str:
        # Convert distance to similarity score (0-100%)
        similarity = (1 - distance) * 100
        return f"This product matches your criteria with {similarity:.1f}% relevance."

    async def add_product(self, product: Dict) -> None:
        """Add a new product to the vector database"""
        metadata = {
            "name": product["name"],
            "price": float(product["price"]),
            "features": product.get("features", ""),
            "category": product.get("metadata", {}).get("category", ""),
            "brand": product.get("metadata", {}).get("brand", ""),
            **{k: v for k, v in product.get("metadata", {}).items() if k not in ["category", "brand"]},
        }

        self.collection.add(ids=[product["id"]], documents=[product["description"]], metadatas=[metadata])

    async def debug_print_collection(self):
        """Debug method to print collection contents"""
        try:
            all_data = self.collection.get()
            print("\nDebug - Collection contents:")
            print(f"Total items: {len(all_data['ids'])}")
            print("Sample items:")
            for i in range(min(3, len(all_data["ids"]))):
                print(f"\nID: {all_data['ids'][i]}")
                print(f"Metadata: {all_data['metadatas'][i]}")
                print(f"Document: {all_data['documents'][i][:100]}...")  # First 100 chars
        except Exception as e:
            print(f"Debug - Error printing collection: {str(e)}")
