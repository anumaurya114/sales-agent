import asyncio
import logging

from app.services.product_database import ProductDatabase

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def init_database():
    logger.debug("Starting database initialization...")
    db = ProductDatabase()
    added_count = 0

    # Sample products
    products = [
        {
            "id": "laptop-1",
            "name": "ProBook X1",
            "price": 999.99,
            "description": "High-performance laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": '16GB RAM\n512GB SSD\nIntel i7\n15.6" 4K Display',
            "metadata": {"category": "laptops", "brand": "ProBook", "in_stock": True},
        },
        {
            "id": "laptop-2",
            "name": "EliteBook X2",
            "price": 1099.99,
            "description": "Ultra-portable laptop with 8GB RAM, 256GB SSD, and Intel i5 processor",
            "features": '8GB RAM\n256GB SSD\nIntel i5\n13.3" FHD Display',
            "metadata": {"category": "laptops", "brand": "EliteBook", "in_stock": True},
        },
        {
            "id": "laptop-3",
            "name": "Spectre X3",
            "price": 1199.99,
            "description": "Premium laptop with 16GB RAM, 1TB SSD, and Intel i9 processor",
            "features": '16GB RAM\n1TB SSD\nIntel i9\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "Spectre", "in_stock": True},
        },
        {
            "id": "laptop-4",
            "name": "Pavilion X4",
            "price": 899.99,
            "description": "Affordable laptop with 8GB RAM, 256GB SSD, and Intel i3 processor",
            "features": '8GB RAM\n256GB SSD\nIntel i3\n14" HD Display',
            "metadata": {"category": "laptops", "brand": "Pavilion", "in_stock": True},
        },
        {
            "id": "laptop-5",
            "name": "Omen X5",
            "price": 1299.99,
            "description": "Gaming laptop with 32GB RAM, 2TB SSD, and Intel i9 processor",
            "features": '32GB RAM\n2TB SSD\nIntel i9\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "Omen", "in_stock": True},
        },
        {
            "id": "laptop-6",
            "name": "Envy X6",
            "price": 999.99,
            "description": "Convertible laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": '16GB RAM\n512GB SSD\nIntel i7\n13.3" FHD Display',
            "metadata": {"category": "laptops", "brand": "Envy", "in_stock": True},
        },
        {
            "id": "laptop-7",
            "name": "ZBook X7",
            "price": 1499.99,
            "description": "Workstation laptop with 64GB RAM, 4TB SSD, and Intel Xeon processor",
            "features": '64GB RAM\n4TB SSD\nIntel Xeon\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "ZBook", "in_stock": True},
        },
        {
            "id": "laptop-8",
            "name": "Spectre X8",
            "price": 1699.99,
            "description": "Premium laptop with 32GB RAM, 2TB SSD, and Intel i9 processor",
            "features": '32GB RAM\n2TB SSD\nIntel i9\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "Spectre", "in_stock": True},
        },
        {
            "id": "laptop-9",
            "name": "Pavilion X9",
            "price": 799.99,
            "description": "Affordable laptop with 8GB RAM, 256GB SSD, and Intel i3 processor",
            "features": '8GB RAM\n256GB SSD\nIntel i3\n14" HD Display',
            "metadata": {"category": "laptops", "brand": "Pavilion", "in_stock": True},
        },
        {
            "id": "laptop-10",
            "name": "Omen X10",
            "price": 1399.99,
            "description": "Gaming laptop with 32GB RAM, 1TB SSD, and Intel i9 processor",
            "features": '32GB RAM\n1TB SSD\nIntel i9\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "Omen", "in_stock": True},
        },
        {
            "id": "laptop-11",
            "name": "Envy X11",
            "price": 1099.99,
            "description": "Convertible laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": '16GB RAM\n512GB SSD\nIntel i7\n13.3" FHD Display',
            "metadata": {"category": "laptops", "brand": "Envy", "in_stock": True},
        },
        {
            "id": "laptop-12",
            "name": "ZBook X12",
            "price": 1599.99,
            "description": "Workstation laptop with 64GB RAM, 4TB SSD, and Intel Xeon processor",
            "features": '64GB RAM\n4TB SSD\nIntel Xeon\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "ZBook", "in_stock": True},
        },
        {
            "id": "laptop-13",
            "name": "Spectre X13",
            "price": 1799.99,
            "description": "Premium laptop with 32GB RAM, 2TB SSD, and Intel i9 processor",
            "features": '32GB RAM\n2TB SSD\nIntel i9\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "Spectre", "in_stock": True},
        },
        {
            "id": "laptop-14",
            "name": "Pavilion X14",
            "price": 899.99,
            "description": "Affordable laptop with 8GB RAM, 256GB SSD, and Intel i3 processor",
            "features": '8GB RAM\n256GB SSD\nIntel i3\n14" HD Display',
            "metadata": {"category": "laptops", "brand": "Pavilion", "in_stock": True},
        },
        {
            "id": "laptop-15",
            "name": "Omen X15",
            "price": 1499.99,
            "description": "Gaming laptop with 32GB RAM, 1TB SSD, and Intel i9 processor",
            "features": '32GB RAM\n1TB SSD\nIntel i9\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "Omen", "in_stock": True},
        },
        {
            "id": "laptop-16",
            "name": "Envy X16",
            "price": 999.99,
            "description": "Convertible laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": '16GB RAM\n512GB SSD\nIntel i7\n13.3" FHD Display',
            "metadata": {"category": "laptops", "brand": "Envy", "in_stock": True},
        },
        {
            "id": "laptop-17",
            "name": "ZBook X17",
            "price": 1599.99,
            "description": "Workstation laptop with 64GB RAM, 4TB SSD, and Intel Xeon processor",
            "features": '64GB RAM\n4TB SSD\nIntel Xeon\n17.3" 4K Display',
            "metadata": {"category": "laptops", "brand": "ZBook", "in_stock": True},
        },
        # Add these products to the existing products list
        {
            "id": "gaming-laptop-1",
            "name": "MSI Raider GE76",
            "price": 1999.99,
            "description": "High-performance gaming laptop with RTX 3080, 16GB RAM, and advanced cooling",
            "features": '16GB RAM\nRTX 3080\n1TB SSD\n17.3" 360Hz Display\nIntel i9\nRGB Keyboard',
            "metadata": {"category": "gaming laptops", "brand": "MSI", "in_stock": True},
        },
        {
            "id": "gaming-laptop-2",
            "name": "MSI Stealth GS66",
            "price": 1699.99,
            "description": "Slim gaming laptop with RTX 3070, 16GB RAM, and excellent battery life",
            "features": '16GB RAM\nRTX 3070\n1TB NVMe SSD\n15.6" 240Hz Display\nIntel i7\nPer-Key RGB',
            "metadata": {"category": "gaming laptops", "brand": "MSI", "in_stock": True},
        },
    ]

    for product in products[:]:
        try:
            await db.add_product(product)
            added_count += 1
            logger.debug(
                f"Added product: {product['name']} (ID: {product['id']}, Brand: {product.get('metadata', {}).get('brand')})"
            )
        except Exception as e:
            logger.error(f"Failed to add product {product['id']}: {str(e)}")

    logger.info(f"Successfully added {added_count} products to the database")

    # Test search immediately after initialization
    logger.debug("\nTesting MSI laptop search after initialization:")
    test_products = await db.search_products(brand="MSI", category="gaming laptops")
    logger.debug(f"Found {len(test_products)} MSI products:")
    for product in test_products:
        logger.debug(f"- {product.name} (Brand: {product.metadata.get('brand')})")


async def test_database():
    logger.debug("\nRunning database test...")
    db = ProductDatabase()

    # Debug print collection contents
    await db.debug_print_collection()

    # Test basic search
    logger.debug("\nTesting search for MSI laptops:")
    products = await db.search_products(brand="MSI", category="gaming laptops")
    logger.debug(f"\nFound {len(products)} products")
    for product in products:
        logger.debug(f"\n- {product.name}")
        logger.debug(f"  Price: ${product.price}")
        logger.debug(f"  Features: {product.features}")
        logger.debug(f"  Category: {product.metadata.get('category')}")
        logger.debug(f"  Brand: {product.metadata.get('brand')}")


if __name__ == "__main__":
    asyncio.run(init_database())
    logger.debug("\nTesting database contents:")
    asyncio.run(test_database())
