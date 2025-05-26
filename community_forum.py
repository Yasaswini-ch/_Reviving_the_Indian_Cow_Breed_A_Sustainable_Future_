import streamlit as st
import json
import os
from datetime import datetime
import uuid # Ensure uuid is imported if used for IDs

# Path relative to where app.py is run
FORUM_POST_FILE_COMMUNITY = "forum_posts_community_module.json"

# @st.cache_resource # Caching posts can be complex with modifications; direct session_state is often simpler
def load_forum_posts_from_file_community(): # Renamed for clarity
    if os.path.exists(FORUM_POST_FILE_COMMUNITY):
        try:
            with open(FORUM_POST_FILE_COMMUNITY, "r", encoding="utf-8") as f:
                posts = json.load(f)
                for i, post_item in enumerate(posts): # Use post_item
                    if "id" not in post_item: # Ensure ID exists
                        post_item["id"] = post_item.get("timestamp", str(uuid.uuid4())) + "_" + str(i)
                    if "replies" not in post_item: # Ensure replies list exists
                        post_item["replies"] = []
                return posts
        except json.JSONDecodeError:
            st.toast(f"Forum file '{FORUM_POST_FILE_COMMUNITY}' seems corrupted. Starting fresh.", icon="⚠️")
            return []
        except Exception as e:
            st.toast(f"Error loading forum posts: {e}", icon="🔥")
            return []
    return []

def save_forum_posts_to_file_community(posts_list_to_save): # Renamed parameter
    try:
        with open(FORUM_POST_FILE_COMMUNITY, "w", encoding="utf-8") as f:
            json.dump(posts_list_to_save, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.toast(f"Error saving forum posts: {e}", icon="🔥")

def render_community_forum():
    # Main title and markdown for the forum page can be set in app.py
    # st.title("💬 Community Forum")
    # st.markdown("Ask questions, share experiences, or help others.")

    # Initialize forum_posts in session state, specific to this module's rendering context
    # Use a unique session_state key to avoid conflicts with app.py's main state
    if "community_module_forum_posts_list" not in st.session_state:
        st.session_state.community_module_forum_posts_list = load_forum_posts_from_file_community()

    # Display messages (success, warning, etc.) specific to forum actions
    if "community_module_user_message" in st.session_state and st.session_state.community_module_user_message:
        message_data = st.session_state.community_module_user_message
        msg_type = message_data.get("type", "info")
        msg_text = message_data.get("text", "")
        if msg_text:
            if msg_type == "success": st.success(msg_text)
            elif msg_type == "warning": st.warning(msg_text)
            elif msg_type == "error": st.error(msg_text)
            else: st.info(msg_text)
        del st.session_state.community_module_user_message

    with st.expander("📢 Create a New Post", expanded=not bool(st.session_state.community_module_forum_posts_list)):
        with st.form("community_module_main_post_form", clear_on_submit=True): # Unique form key
            author_name_input = st.text_input("👤 Your Name", key="community_module_post_author_name",
                                         value=st.session_state.get("username", "") if st.session_state.get("logged_in") else "")
            post_text_input = st.text_area("💬 Your Message", height=150, key="community_module_post_message_text")
            submitted_button = st.form_submit_button("Post Message")

            if submitted_button:
                final_author_name = author_name_input.strip() or "Anonymous"
                if post_text_input.strip():
                    new_post_entry = { # Renamed variable
                        "id": str(uuid.uuid4()), "name": final_author_name, "text": post_text_input.strip(),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "replies": []
                    }
                    st.session_state.community_module_forum_posts_list.insert(0, new_post_entry)
                    save_forum_posts_to_file_community(st.session_state.community_module_forum_posts_list)
                    st.session_state.community_module_user_message = {"type": "success", "text": "✅ Post submitted!"}
                    st.rerun()
                else:
                    st.warning("Message cannot be empty.")

    st.markdown("---")
    st.subheader("📝 Recent Discussions")

    if not st.session_state.community_module_forum_posts_list:
        st.info("No discussions yet. Be the first to start one!")
    else:
        # Iterate over a copy if direct modification in loop, but append to sub-list is generally fine
        for post_data_item in list(st.session_state.community_module_forum_posts_list):
            with st.container(border=True):
                st.markdown(f"**👤 {post_data_item['name']}**    <small>🕒 *{post_data_item['timestamp']}*</small>", unsafe_allow_html=True)
                st.write(post_data_item["text"])

                if post_data_item.get("replies"):
                    for reply_entry in post_data_item["replies"]: # Renamed variable
                        st.markdown(f"   ↪️ **{reply_entry['name']}** (*{reply_entry['timestamp']}*): {reply_entry['text']}")

                with st.expander(f"💬 Reply to {post_data_item['name']}", expanded=False):
                    reply_form_key_unique = f"community_module_reply_form_{post_data_item['id']}" # Unique key
                    with st.form(key=reply_form_key_unique, clear_on_submit=True):
                        reply_author_input = st.text_input("👤 Your Name", key=f"community_module_reply_author_{post_data_item['id']}",
                                                     value=st.session_state.get("username", "") if st.session_state.get("logged_in") else "")
                        reply_text_content_input = st.text_area("💬 Your Reply", height=75, key=f"community_module_reply_text_{post_data_item['id']}")
                        submit_reply_button = st.form_submit_button("Submit Reply")

                        if submit_reply_button:
                            final_reply_author_name = reply_author_input.strip() or "Anonymous"
                            if reply_text_content_input.strip():
                                new_reply_entry = { # Renamed variable
                                    "name": final_reply_author_name, "text": reply_text_content_input.strip(),
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                # Find post and append reply (modifying item in session_state list directly)
                                for p_loop_item in st.session_state.community_module_forum_posts_list: # Renamed loop var
                                    if p_loop_item['id'] == post_data_item['id']:
                                        p_loop_item['replies'].append(new_reply_entry)
                                        break
                                save_forum_posts_to_file_community(st.session_state.community_module_forum_posts_list)
                                st.session_state.community_module_user_message = {"type": "success", "text": "✅ Reply posted!"}
                                st.rerun()
                            else:
                                st.warning("Reply message cannot be empty.")
    st.markdown("<br><hr><sub>Keep discussions respectful and constructive.</sub>", unsafe_allow_html=True)