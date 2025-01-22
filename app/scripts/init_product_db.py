from app.services.product_database import ProductDatabase
import asyncio

async def init_database():
    db = ProductDatabase()
    
    # Sample products
    products = [
        {
            "id": "laptop-1",
            "name": "ProBook X1",
            "price": 999.99,
            "description": "High-performance laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": "16GB RAM\n512GB SSD\nIntel i7\n15.6\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "ProBook",
                "in_stock": True
            }
        },
        {
            "id": "phone-1",
            "name": "SmartPhone Pro",
            "price": 799.99,
            "description": "Latest smartphone with 5G capability, 128GB storage, and advanced camera system",
            "features": "5G\n128GB Storage\nTriple Camera\n6.7\" OLED",
            "metadata": {
                "category": "phones",
                "brand": "Smart",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-2",
            "name": "EliteBook X2",
            "price": 1099.99,
            "description": "Ultra-portable laptop with 8GB RAM, 256GB SSD, and Intel i5 processor",
            "features": "8GB RAM\n256GB SSD\nIntel i5\n13.3\" FHD Display",
            "metadata": {
                "category": "laptops",
                "brand": "EliteBook",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-2",
            "name": "SmartPhone Max",
            "price": 899.99,
            "description": "High-end smartphone with 5G capability, 256GB storage, and advanced camera system",
            "features": "5G\n256GB Storage\nQuad Camera\n6.9\" OLED",
            "metadata": {
                "category": "phones",
                "brand": "Max",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-3",
            "name": "Spectre X3",
            "price": 1199.99,
            "description": "Premium laptop with 16GB RAM, 1TB SSD, and Intel i9 processor",
            "features": "16GB RAM\n1TB SSD\nIntel i9\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "Spectre",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-3",
            "name": "SmartPhone Ultra",
            "price": 999.99,
            "description": "Flagship smartphone with 5G capability, 512GB storage, and advanced camera system",
            "features": "5G\n512GB Storage\nPenta Camera\n6.5\" OLED",
            "metadata": {
                "category": "phones",
                "brand": "Ultra",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-4",
            "name": "Pavilion X4",
            "price": 899.99,
            "description": "Affordable laptop with 8GB RAM, 256GB SSD, and Intel i3 processor",
            "features": "8GB RAM\n256GB SSD\nIntel i3\n14\" HD Display",
            "metadata": {
                "category": "laptops",
                "brand": "Pavilion",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-4",
            "name": "SmartPhone Lite",
            "price": 699.99,
            "description": "Budget smartphone with 4G capability, 64GB storage, and dual camera system",
            "features": "4G\n64GB Storage\nDual Camera\n5.5\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Lite",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-5",
            "name": "Omen X5",
            "price": 1299.99,
            "description": "Gaming laptop with 32GB RAM, 2TB SSD, and Intel i9 processor",
            "features": "32GB RAM\n2TB SSD\nIntel i9\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "Omen",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-5",
            "name": "SmartPhone Mini",
            "price": 599.99,
            "description": "Compact smartphone with 4G capability, 32GB storage, and single camera system",
            "features": "4G\n32GB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Mini",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-6",
            "name": "Envy X6",
            "price": 999.99,
            "description": "Convertible laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": "16GB RAM\n512GB SSD\nIntel i7\n13.3\" FHD Display",
            "metadata": {
                "category": "laptops",
                "brand": "Envy",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-6",
            "name": "SmartPhone Lite Plus",
            "price": 499.99,
            "description": "Entry-level smartphone with 4G capability, 16GB storage, and single camera system",
            "features": "4G\n16GB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Lite Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-7",
            "name": "ZBook X7",
            "price": 1499.99,
            "description": "Workstation laptop with 64GB RAM, 4TB SSD, and Intel Xeon processor",
            "features": "64GB RAM\n4TB SSD\nIntel Xeon\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "ZBook",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-7",
            "name": "SmartPhone Ultra Plus",
            "price": 399.99,
            "description": "Basic smartphone with 4G capability, 8GB storage, and single camera system",
            "features": "4G\n8GB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Ultra Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-8",
            "name": "Spectre X8",
            "price": 1699.99,
            "description": "Premium laptop with 32GB RAM, 2TB SSD, and Intel i9 processor",
            "features": "32GB RAM\n2TB SSD\nIntel i9\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "Spectre",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-8",
            "name": "SmartPhone Max Plus",
            "price": 299.99,
            "description": "Entry-level smartphone with 4G capability, 4GB storage, and single camera system",
            "features": "4G\n4GB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Max Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-9",
            "name": "Pavilion X9",
            "price": 799.99,
            "description": "Affordable laptop with 8GB RAM, 256GB SSD, and Intel i3 processor",
            "features": "8GB RAM\n256GB SSD\nIntel i3\n14\" HD Display",
            "metadata": {
                "category": "laptops",
                "brand": "Pavilion",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-9",
            "name": "SmartPhone Lite Plus",
            "price": 199.99,
            "description": "Basic smartphone with 4G capability, 2GB storage, and single camera system",
            "features": "4G\n2GB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Lite Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-10",
            "name": "Omen X10",
            "price": 1399.99,
            "description": "Gaming laptop with 32GB RAM, 1TB SSD, and Intel i9 processor",
            "features": "32GB RAM\n1TB SSD\nIntel i9\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "Omen",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-10",
            "name": "SmartPhone Mini Plus",
            "price": 99.99,
            "description": "Entry-level smartphone with 4G capability, 1GB storage, and single camera system",
            "features": "4G\n1GB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Mini Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-11",
            "name": "Envy X11",
            "price": 1099.99,
            "description": "Convertible laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": "16GB RAM\n512GB SSD\nIntel i7\n13.3\" FHD Display",
            "metadata": {
                "category": "laptops",
                "brand": "Envy",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-11",
            "name": "SmartPhone Lite Plus",
            "price": 49.99,
            "description": "Basic smartphone with 4G capability, 512MB storage, and single camera system",
            "features": "4G\n512MB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Lite Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-12",
            "name": "ZBook X12",
            "price": 1599.99,
            "description": "Workstation laptop with 64GB RAM, 4TB SSD, and Intel Xeon processor",
            "features": "64GB RAM\n4TB SSD\nIntel Xeon\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "ZBook",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-12",
            "name": "SmartPhone Ultra Plus",
            "price": 19.99,
            "description": "Entry-level smartphone with 4G capability, 256MB storage, and single camera system",
            "features": "4G\n256MB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Ultra Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-13",
            "name": "Spectre X13",
            "price": 1799.99,
            "description": "Premium laptop with 32GB RAM, 2TB SSD, and Intel i9 processor",
            "features": "32GB RAM\n2TB SSD\nIntel i9\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "Spectre",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-13",
            "name": "SmartPhone Max Plus",
            "price": 9.99,
            "description": "Basic smartphone with 4G capability, 128MB storage, and single camera system",
            "features": "4G\n128MB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Max Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-14",
            "name": "Pavilion X14",
            "price": 899.99,
            "description": "Affordable laptop with 8GB RAM, 256GB SSD, and Intel i3 processor",
            "features": "8GB RAM\n256GB SSD\nIntel i3\n14\" HD Display",
            "metadata": {
                "category": "laptops",
                "brand": "Pavilion",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-14",
            "name": "SmartPhone Lite Plus",
            "price": 4.99,
            "description": "Basic smartphone with 4G capability, 64MB storage, and single camera system",
            "features": "4G\n64MB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Lite Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-15",
            "name": "Omen X15",
            "price": 1499.99,
            "description": "Gaming laptop with 32GB RAM, 1TB SSD, and Intel i9 processor",
            "features": "32GB RAM\n1TB SSD\nIntel i9\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "Omen",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-15",
            "name": "SmartPhone Mini Plus",
            "price": 2.99,
            "description": "Entry-level smartphone with 4G capability, 32MB storage, and single camera system",
            "features": "4G\n32MB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Mini Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-16",
            "name": "Envy X16",
            "price": 999.99,
            "description": "Convertible laptop with 16GB RAM, 512GB SSD, and Intel i7 processor",
            "features": "16GB RAM\n512GB SSD\nIntel i7\n13.3\" FHD Display",
            "metadata": {
                "category": "laptops",
                "brand": "Envy",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "phone-16",
            "name": "SmartPhone Lite Plus",
            "price": 1.99,
            "description": "Basic smartphone with 4G capability, 16MB storage, and single camera system",
            "features": "4G\n16MB Storage\nSingle Camera\n5.0\" LCD",
            "metadata": {
                "category": "phones",
                "brand": "Lite Plus",
                "in_stock": True
            }
        },
        # Add more products as needed
        {
            "id": "laptop-17",
            "name": "ZBook X17",
            "price": 1599.99,
            "description": "Workstation laptop with 64GB RAM, 4TB SSD, and Intel Xeon processor",
            "features": "64GB RAM\n4TB SSD\nIntel Xeon\n17.3\" 4K Display",
            "metadata": {
                "category": "laptops",
                "brand": "ZBook",
                "in_stock": True
            }
        }
    ]
    
    for product in products:
        await db.add_product(product)

if __name__ == "__main__":
    asyncio.run(init_database())