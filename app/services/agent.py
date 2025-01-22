import json
from enum import Enum
from typing import Dict, Optional

from openai import OpenAI

from app.config import settings
from app.models.conversation import Conversation
import re
from app.services.product_database import ProductDatabase


class ActionType(Enum):
    WELCOME = "welcome"
    UNDERSTAND_NEEDS = "understand_needs"
    GATHER_REQUIREMENTS = "gather_requirements"
    SEARCH_PRODUCTS = "search_products"
    PROVIDE_RECOMMENDATIONS = "provide_recommendations"
    HANDLE_QUESTIONS = "handle_questions"
    CONFIRM_SELECTION = "confirm_selection"
    END_CONVERSATION = "end_conversation"


class SalesAgent:
    def __init__(self, conversation: Conversation=None):
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=api_key)
        self.current_state = ActionType.WELCOME
        self.tools = self._create_tools()
        self.db = ProductDatabase()

    def _create_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "determine_next_action",
                    "description": "Determine the next action in the sales conversation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [action.value for action in ActionType],
                                "description": "The next action to take in the conversation",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation for why this action was chosen",
                            },
                            "required_info": {
                                "type": "object",
                                "properties": {
                                    "product_type": {"type": "string"},
                                    "budget": {"type": "string"},
                                    "preferences": {"type": "array", "items": {"type": "string"}},
                                    "constraints": {"type": "array", "items": {"type": "string"}},
                                },
                                "description": "Basic information gathered or needed for the action",
                            },
                            "search_params": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "price_min": {"type": "number"},
                                    "price_max": {"type": "number"},
                                    "brand": {"type": "string"},
                                    "category": {"type": "string"},
                                    "features": {"type": "array", "items": {"type": "string"}},
                                },
                                "description": "Parameters specific to product search",
                            },
                            "recommendation_params": {
                                "type": "object",
                                "properties": {
                                    "use_case": {"type": "string"},
                                    "priority_features": {"type": "array", "items": {"type": "string"}},
                                    "similar_to_product": {"type": "string"},
                                    "exclude_brands": {"type": "array", "items": {"type": "string"}},
                                },
                                "description": "Parameters specific to product recommendations",
                            },
                            "selection_params": {
                                "type": "object",
                                "properties": {
                                    "selected_product_id": {"type": "string"},
                                    "quantity": {"type": "integer"},
                                    "customization_options": {"type": "object"},
                                    "delivery_preferences": {"type": "string"},
                                },
                                "description": "Parameters specific to product selection",
                            },
                        },
                        "required": ["action", "reasoning"],
                    },
                },
            }
        ]

    async def process_message(
        self, conversation: Conversation, message: str, selected_product: Optional[Dict] = None
    ):
        """Process an incoming message and return a response"""
        # Add user message to conversation
        conversation.add_message("user", message)

        # Determine next action
        next_action = await self._determine_next_action(conversation)

        # Execute action and get response
        try:
            # Execute action and await the response
            response = await self._execute_action(next_action, conversation)
            
            # Ensure we have a string response
            if not isinstance(response, str):
                print(f"Warning: Response is not a string, got {type(response)}")
                response = str(response)

            # Add assistant's response to conversation
            conversation.add_message("assistant", response)

            # Return both response and products if available
            return response, getattr(conversation, "last_products", [])
            
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            # Provide a fallback response
            fallback_response = (
                "I apologize, but I encountered an error. "
                "Could you please rephrase your request or try again?"
            )
            conversation.add_message("assistant", fallback_response)
            return fallback_response, []

    def _create_prompt_tools(self) -> list:
        """Create tools for prompt generation"""
        return [{
            "type": "function",
            "function": {
                "name": "generate_prompt_response",
                "description": "Generate a formatted response for different conversation stages",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The formatted response content"
                        }
                    },
                    "required": ["content"]
                }
            }
        }]

    async def _generate_prompt(self, prompt_type: str, context: Dict = None) -> str:
        """Generate dynamic prompts using LLM based on context"""
        # First, format the context to ensure it's clean
        formatted_context = json.dumps(context) if context else ""
        
        system_prompts = {
            "welcome": (
                "You are a helpful sales assistant. Generate a warm, friendly welcome message that: "
                "1. Introduces yourself as an AI shopping assistant "
                "2. Explains how you can help find the perfect laptop "
                "3. Lists 3-4 key ways users can interact with you "
                "4. Asks an opening question about their needs "
                "Keep the tone professional but friendly. Format with emojis and bullet points."
            ),
            "understand_needs": (
                "You are a helpful sales assistant. Generate a response that helps understand the customer's needs. "
                "Consider any previous responses in the context. "
                "Your response should: "
                "1. Acknowledge any information they've already shared "
                "2. Ask focused questions about their primary use case "
                "3. Inquire about key preferences (performance, portability, etc.) "
                "4. Mention budget if not already discussed "
                "Keep the tone conversational and use follow-up questions based on their previous responses. "
                "Format with clear spacing and bullet points where appropriate."
            ),
            "selection": (
                "You are a helpful sales assistant. Generate a response for a customer who has selected a product. "
                "Context: {product_name} has been selected. "
                "Include: "
                "1. Confirmation of their selection "
                "2. Next steps (purchase, compare, more details) "
                "3. A question about their decision "
                "Keep it helpful and provide clear options."
            ),
            "end_conversation": (
                "You are a helpful sales assistant. Generate a friendly conversation closing that: "
                "1. Thanks the customer for their time "
                "2. Summarizes any key decisions made "
                "3. Provides clear next steps "
                "4. Invites them to return if they need more help "
                "Keep it concise and positive."
            )
        }
        try:
            user_message = (
                f"Generate a response for {prompt_type}. "
                f"Format it naturally with appropriate spacing and structure."
            )
            if context:
                user_message += f" Consider this context: {formatted_context}"

            completion = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompts.get(prompt_type, "")},
                    {"role": "user", "content": user_message}
                ],
                tools=self._create_prompt_tools(),
                tool_choice={"type": "function", "function": {"name": "generate_prompt_response"}}
            )

            tool_call = completion.choices[0].message.tool_calls[0]
            
            try:
                # Clean the JSON string before parsing
                clean_arguments = tool_call.function.arguments.strip()
                # Handle the case where the string might start with a newline
                if clean_arguments.startswith('\n'):
                    clean_arguments = clean_arguments[1:]
                
                # Use a more permissive JSON parser
                response_args = json.loads(
                    clean_arguments,
                    strict=False  # This makes the parser more forgiving
                )
                return response_args.get("content", "").strip()
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw arguments: {tool_call.function.arguments}")
                
                # Fallback: Try to extract content directly if JSON parsing fails
                if '"content":' in tool_call.function.arguments:
                    try:
                        # Simple string extraction as fallback
                        content_start = tool_call.function.arguments.index('"content":') + 10
                        content = tool_call.function.arguments[content_start:]
                        content = content.strip().strip('{}"').strip()
                        return content
                    except Exception:
                        pass
                
                return "I apologize, but I encountered an error processing the response. How can I assist you today?"
                
        except Exception as e:
            print(f"Error in _generate_prompt: {str(e)}")
            raise

    async def _determine_next_action(self, conversation: Conversation) -> Dict:
        system_message = f"""
        You are an intelligent sales assistant. Current state: {self.current_state.value}
        
        Analyze the conversation history and determine the next best action.
        Follow this conversation flow:
        1. Start with WELCOME for new users
        2. Move to UNDERSTAND_NEEDS to gather basic requirements
        3. Use GATHER_REQUIREMENTS to get specific details
        4. Only proceed to SEARCH_PRODUCTS when you have:
           - Purpose/use case of the laptop
           - Budget range (if any)
           - Any specific requirements
        
        When user mentions specific product requirements:
        - Extract product type, brand, features, and price range
        - Use SEARCH_PRODUCTS action only when sufficient information is gathered
        - Include these details in search_params
        
        Consider:
        1. What information has been gathered?
        2. What information is still needed?
        3. Is it time to make recommendations?
        4. Does the user need clarification?
        5. Should the conversation end?
        
        Available actions:
        {[action.value for action in ActionType]}
        """

        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                *[{"role": m.role, "content": m.content} for m in conversation.messages],
            ],
            tools=self.tools,
            tool_choice={"type": "function", "function": {"name": "determine_next_action"}},
        )

        tool_call = completion.choices[0].message.tool_calls[0]
        action_args = json.loads(tool_call.function.arguments)

        # For initial messages or when requirements aren't clear, prioritize understanding needs
        if len(conversation.messages) <= 2:
            action_args["action"] = ActionType.UNDERSTAND_NEEDS.value
            return action_args

        # Check if we have enough information to search
        has_requirements = False
        message_content = " ".join(m.content.lower() for m in conversation.messages if m.role == "user")
        
        # Check for basic requirements in the conversation
        has_use_case = any(keyword in message_content for keyword in ["use", "need", "looking for", "want"])
        has_budget = any(keyword in message_content for keyword in ["budget", "price", "cost", "$"])
        has_preferences = any(keyword in message_content for keyword in ["prefer", "must have", "should have", "want"])

        if has_use_case and (has_budget or has_preferences):
            has_requirements = True

        # Override action if requirements aren't met
        if not has_requirements and action_args["action"] == ActionType.SEARCH_PRODUCTS.value:
            action_args["action"] = ActionType.GATHER_REQUIREMENTS.value
            action_args["reasoning"] = "Need to gather more information about user requirements"

        self.current_state = ActionType(action_args["action"])
        return action_args

    async def _execute_action(self, next_action: dict, conversation: Conversation) -> str:
        action = next_action.get("action")
        print("action is", action)
        print("", next_action.get("reasoning"))
        print("", next_action.get("required_info", {}))
        print(
            "",
            "other args",
            {k: v for k, v in next_action.items() if k not in ["action", "reasoning", "required_info"]},
        )

        required_info = next_action.get("required_info", {})
        search_params = next_action.get("search_params", {})
        recommendation_params = next_action.get("recommendation_params", {})
        selection_params = next_action.get("selection_params", {})

        print(
            "",
            "search params",
            search_params,
            "recommendation params",
            recommendation_params,
            "selection params",
            selection_params,
        )

        try:
            if action == "welcome":
                return await self._handle_welcome()
            elif action == "understand_needs":
                print("understanding needs")
                return await self._handle_understand_needs(conversation=conversation)
            elif action == "gather_requirements":
                print("gathering requirements")
                return await self._handle_understand_needs(conversation=conversation)
            elif action == "search_products":
                print("searching products")
                response, products = await self._handle_product_search(
                    required_info, conversation, search_params=search_params
                )
                conversation.last_products = products  # Store products in conversation
                return response
            elif action == "provide_recommendations":
                print("providing recommendations")
                response, products = await self._handle_recommendations(
                    required_info, conversation, recommendation_params=recommendation_params
                )
                conversation.last_products = products  # Store products in conversation
                return response
            elif action == "handle_questions":
                print("handling questions")
                return await self._handle_questions(required_info)
            elif action == "confirm_selection":
                print("confirming selection")
                return await self._handle_selection(required_info, selection_params=selection_params)
            elif action == "end_conversation":
                print("ending conversation")
                return await self._handle_end_conversation()
            else:
                return "I'm not sure how to handle that action. Could you please rephrase your request?"
        except Exception as e:
            print(f"Error executing action {action}: {str(e)}")
            # Provide a fallback response
            return "I apologize, but I encountered an error. Could you please try rephrasing your request?"

    async def _handle_product_search(self, required_info: dict, conversation: Conversation, search_params: dict = None) -> tuple[str, list]:
        if not search_params:
            search_params = {}
        
        # Intelligently build search parameters from both required_info and search_params
        search_criteria = {
            "query": search_params.get("query"),
            "product_type": required_info.get("product_type") or search_params.get("product_type"),
            "min_price": float(required_info.get("budget", "0").replace("$", "").split("-")[0]) if required_info.get("budget") else search_params.get("price_min"),
            "max_price": float(required_info.get("budget", "0").replace("$", "").split("-")[-1]) if required_info.get("budget") else search_params.get("price_max"),
            "preferences": list(set(required_info.get("preferences", []) + search_params.get("features", []))),
            "constraints": required_info.get("constraints", []),
            "brand": search_params.get("brand"),
            "category": search_params.get("category"),
            "features": search_params.get("features", [])
        }

        try:
            products = await self.db.search_products(**search_criteria)

            if not products:
                return (
                    "I couldn't find any products matching your criteria. Would you like to adjust your requirements?",
                    [],
                )

            # Format response with product details
            response = "Based on your requirements, here are some products that match your criteria:\n\n"
            for product in products:
                response += f"🔍 {product.name}\n"
                response += f"💰 Price: ${product.price}\n"
                response += f"✨ Features: {product.features}\n"
                response += f"📝 Description: {product.description}\n"
                response += f"{product.match_explanation}\n\n"

            response += "\nWould you like more specific details about any of these products? Or shall we refine the search further?"
            return response, products

        except Exception as e:
            print(f"Error during product search: {str(e)}")
            return "I encountered an error while searching for products. Please try again with different criteria.", []

    async def _handle_recommendations(self, required_info: dict, conversation: Conversation, recommendation_params: dict = None) -> tuple[str, list]:
        if not recommendation_params:
            recommendation_params = {}

        # Build comprehensive recommendation criteria
        recommendation_criteria = {
            "product_type": required_info.get("product_type"),
            "preferences": list(set(required_info.get("preferences", []) + recommendation_params.get("priority_features", []))),
            "constraints": required_info.get("constraints", []),
            "use_case": recommendation_params.get("use_case"),
            "similar_to_product": recommendation_params.get("similar_to_product"),
            "exclude_brands": recommendation_params.get("exclude_brands", []),
            "budget_range": required_info.get("budget")
        }

        try:
            products = await self.db.get_recommendations(**recommendation_criteria)
            
            if not products:
                alternative_products = await self._find_alternative_recommendations(recommendation_criteria)
                if alternative_products:
                    return (
                        "I couldn't find exact matches, but here are some similar alternatives that might interest you:\n\n"
                        + self._format_product_list(alternative_products),
                        alternative_products
                    )

            # Format response with product details
            response = "Here are some recommendations based on your preferences:\n\n"
            for product in products:
                response += f"🔍 {product.name}\n"
                response += f"💰 Price: ${product.price}\n"
                response += f"✨ Features: {product.features}\n"
                response += f"📝 Description: {product.description}\n"
                response += f"{product.match_explanation}\n\n"

            response += "\nWould you like more details about any of these products?"
            return response, products

        except Exception as e:
            print(f"Error during recommendations: {str(e)}")
            return "I encountered an error while generating recommendations. Please try again.", []


    async def _handle_questions(self, required_info: Dict):
        # Get the selected product from the conversation context
        selected_product = required_info.get("selected_product")
        
        if not selected_product:
            return "I'm not sure which product you're asking about. Could you please specify which product you'd like to know more about?"

        # Get the aspect the user is asking about
        aspect = selected_product.get("aspect", "general")
        product_id = selected_product.get("product_id")

        # Format response based on the aspect
        if aspect == "price":
            return f"The {selected_product.get('name')} is priced at ${selected_product.get('price')}. " \
                   "This price reflects its high-end specifications and build quality."
        elif aspect == "features":
            features = selected_product.get('features', [])
            return f"The key features of {selected_product.get('name')} include:\n" + \
                   "\n".join([f"• {feature}" for feature in features])
        elif aspect == "specs":
            return f"Here are the detailed specifications of {selected_product.get('name')}:\n" + \
                   f"{selected_product.get('description')}\n" + \
                   f"Features: {selected_product.get('features')}"
        else:
            # General information about the product
            return f"The {selected_product.get('name')} is a {selected_product.get('metadata', {}).get('category', '')} " \
                   f"from {selected_product.get('metadata', {}).get('brand', '')}.\n" \
                   f"Price: ${selected_product.get('price')}\n" \
                   f"Description: {selected_product.get('description')}\n" \
                   f"Key Features: {selected_product.get('features')}\n\n" \
                   "Would you like to know more about any specific aspect of this product?"

    async def _handle_selection(self, required_info: Dict, selection_params: Dict):
        """Handle product selection with dynamic response"""
        selected_product = required_info.get("selected_product", {})
        if not selected_product:
            return "I couldn't find the product you selected. Could you please try again?"

        product_name = selected_product.get("name", "the product")
        price = selected_product.get("price", 0)
        
        context = {
            "product_name": product_name,
            "price": price,
            "action": selection_params.get("action", "view"),
            "comparison": selection_params.get("compare_with", []),
        }

        try:
            # Generate dynamic selection response
            response = await self._generate_prompt("selection", context)
            
            # Add specific product details if available
            if selected_product.get("features"):
                response += f"\n\nKey Features of {product_name}:\n"
                for feature in selected_product["features"].split("\n"):
                    response += f"• {feature}\n"
            
            # Add price and availability
            response += f"\nPrice: ${price:,.2f}\n"
            if selected_product.get("metadata", {}).get("in_stock"):
                response += "✅ In Stock\n"
            
            # Add call to action
            response += "\nWould you like to:\n"
            response += "• Proceed with this selection\n"
            response += "• Compare with other models\n"
            response += "• See more details\n"
            response += "• Look for other options"
            
            return response
        except Exception as e:
            print(f"Error generating selection response: {str(e)}")
            return "I've noted your selection. Would you like to proceed with this product or see more options?"

    async def _handle_end_conversation(self):
        """Handle end of conversation with dynamic response"""
        try:
            # Get conversation summary context (could be expanded)
            context = {
                "selected_products": getattr(self, "selected_products", []),
                "preferences_gathered": bool(getattr(self, "customer_preferences", {})),
                "search_completed": bool(getattr(self, "last_search_results", [])),
            }
            
            # Generate dynamic ending message
            response = await self._generate_prompt("end_conversation", context)
            
            # Reset conversation state
            self.current_state = ActionType.WELCOME
            
            return response
        except Exception as e:
            print(f"Error generating end conversation response: {str(e)}")
            return (
                "Thank you for shopping with us! If you need any further assistance, "
                "feel free to start a new conversation. Have a great day! 👋"
            )

    # Update other handlers to use dynamic prompts
    async def _handle_welcome(self):
        try:
            return await self._generate_prompt("welcome")
        except Exception as e:
            print(f"Error generating welcome prompt: {str(e)}")
            raise

    async def _handle_understand_needs(self, conversation: Conversation):
        """Handle understanding needs with live product suggestions."""
        try:
            # Analyze conversation for existing information
            conversation_text = " ".join(m.content.lower() for m in conversation.messages)
            current_info = self._extract_current_requirements(conversation_text)
            
            # Format the query string properly
            query_terms = []
            if current_info.get("keywords"):
                query_terms.extend(current_info["keywords"])
            if "laptop" not in query_terms:  # Add laptop if not present
                query_terms.append("laptop")
            
            query = " ".join(query_terms) if query_terms else "laptop"  # Ensure query is a string
            
            # Get filtered products based on current information
            filtered_products = await self.db.search_products(
                query=query,  # Pass the properly formatted query string
                min_price=current_info.get("min_price"),
                max_price=current_info.get("max_price"),
                preferences=current_info.get("preferences", []),
                n_results=3  # Show fewer products at this early stage
            )

            # Generate base response using the prompt template
            context = {
                "previous_responses": [m.content for m in conversation.messages if m.role == "user"],
                "current_state": self.current_state.value,
                "current_understanding": current_info,
                "found_products": bool(filtered_products)
            }
            
            base_response = await self._generate_prompt("understand_needs", context)

            # Add product suggestions if any were found
            if filtered_products:
                product_section = "\n\nBased on what you've mentioned, here are some initial suggestions:\n\n"
                for product in filtered_products:
                    product_section += f"💻 {product.name}\n"
                    product_section += f"   💰 ${product.price:,.2f}\n"
                    if product.features:
                        top_features = (
                            product.features[:2] 
                            if isinstance(product.features, list) 
                            else product.features.split('\n')[:2]
                        )
                        product_section += f"   ✨ Highlights: {', '.join(top_features)}\n"
                    product_section += "\n"
                
                product_section += "\nThese are just initial matches. Let me ask you a few questions to find the perfect laptop for you.\n"
                response = base_response + product_section
            else:
                response = base_response

            # Store products in conversation for the main.py response
            conversation.last_products = filtered_products
            
            return response

        except Exception as e:
            print(f"Error in _handle_understand_needs: {str(e)}")
            # Log the full error details for debugging
            import traceback
            print(f"Full error traceback:\n{traceback.format_exc()}")
            return "Could you tell me more about what you're looking for in a laptop?"

    async def process_message(
        self, conversation: Conversation, message: str, selected_product: Optional[Dict] = None
    ):
        """Process an incoming message and return a response"""
        # Add user message to conversation
        conversation.add_message("user", message)

        # Determine next action
        next_action = await self._determine_next_action(conversation)

        # Execute action and get response
        try:
            # Execute action and await the response
            response = await self._execute_action(next_action, conversation)
            
            # Ensure we have a string response
            if not isinstance(response, str):
                print(f"Warning: Response is not a string, got {type(response)}")
                response = str(response)

            # Add assistant's response to conversation
            conversation.add_message("assistant", response)

            # Return both response and products
            # If no products were set during action execution, return empty list
            return response, getattr(conversation, "last_products", [])
            
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            fallback_response = (
                "I apologize, but I encountered an error. "
                "Could you please rephrase your request or try again?"
            )
            conversation.add_message("assistant", fallback_response)
            return fallback_response, []

    def _extract_current_requirements(self, conversation_text: str) -> dict:
        """Extract current requirements from conversation text."""
        requirements = {
            "keywords": [],
            "min_price": None,
            "max_price": None,
            "preferences": []
        }
        
        # Extract price ranges
        price_matches = re.findall(r'\$?\d+(?:,\d+)?(?:\s*-\s*\$?\d+(?:,\d+)?)?k?\s*(?:usd|dollars)?', conversation_text, re.IGNORECASE)
        if price_matches:
            # Process the first found price range
            price_range = price_matches[0].lower().replace('$', '').replace(',', '').replace('usd', '').replace('dollars', '').strip()
            if 'k' in price_range:
                price_range = price_range.replace('k', '000')
            if '-' in price_range:
                min_str, max_str = price_range.split('-')
                requirements['min_price'] = float(min_str.strip())
                requirements['max_price'] = float(max_str.strip())
            else:
                if 'under' in conversation_text or 'less than' in conversation_text:
                    requirements['max_price'] = float(price_range.strip())
                else:
                    requirements['min_price'] = float(price_range.strip())
        
        # Extract specific hardware requirements
        hardware_specs = {
            'ram': r'(\d+)\s*gb\s*ram',
            'processor': r'(intel|amd|ryzen|i[3579])',
            'graphics': r'(nvidia|rtx|gtx|geforce)',
            'storage': r'(\d+)\s*(?:gb|tb)\s*(?:ssd|hdd)',
        }
        
        for spec_type, pattern in hardware_specs.items():
            matches = re.findall(pattern, conversation_text, re.IGNORECASE)
            if matches:
                requirements['preferences'].append(f"{spec_type}:{matches[0]}")
        
        # Extract general preferences
        common_features = {
            'performance': ['gaming', 'high-end', 'powerful', 'fast'],
            'portability': ['portable', 'lightweight', 'thin', 'slim'],
            'display': ['screen', 'display', 'resolution', '4k', 'hdr'],
            'battery': ['battery', 'long-lasting', 'all-day'],
            'build': ['premium', 'professional', 'durable', 'quality']
        }
        
        for category, keywords in common_features.items():
            if any(keyword in conversation_text for keyword in keywords):
                requirements['preferences'].append(category)
        
        # Extract general keywords for search
        words = conversation_text.split()
        requirements['keywords'] = [word for word in words 
                                  if len(word) > 3 
                                  and word not in set(sum(common_features.values(), []))
                                  and not any(char.isdigit() for char in word)][:5]
        
        return requirements

    def _determine_missing_info(self, current_info: dict) -> list:
        """Determine what important information is still missing."""
        missing = []
        
        if not current_info.get('min_price') and not current_info.get('max_price'):
            missing.append('budget')
        
        if len(current_info.get('preferences', [])) < 2:
            missing.append('preferences')
        
        if len(current_info.get('keywords', [])) < 2:
            missing.append('use_case')
        
        return missing

    def _generate_requirement_question(self, missing_type: str) -> str:
        """Generate a natural question for the missing requirement type."""
        questions = {
            'budget': [
                "What's your budget range for this purchase?",
                "How much are you looking to invest in this product?",
                "What price range are you comfortable with?"
            ],
            'preferences': [
                "What features are most important to you?",
                "Do you have any specific requirements or preferences?",
                "What qualities are you looking for in this product?"
            ],
            'use_case': [
                "How do you plan to use this product?",
                "What will be the primary purpose of this product?",
                "What kind of tasks do you need this product for?"
            ]
        }
        
        return random.choice(questions.get(missing_type, ["Could you tell me more about what you're looking for?"]))