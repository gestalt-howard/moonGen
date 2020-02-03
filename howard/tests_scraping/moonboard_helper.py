from selenium import webdriver
from selenium_helper import *

import os
import pickle

def click_login_area(driver):
    login_elem = None
    a_elems = driver.find_elements_by_tag_name('a')
    for a in a_elems:
        if a.text=='LOGIN/REGISTER':
            login_elem = a
            break
    if login_elem==None:
        print('Failed to find Login Button')
    else:
        login_elem.click()
    return login_elem

def click_relogin(driver):
    return find_and_click(driver, 'a', 'title', 'Go to dashboard')

def click_login_button(driver):
    return find_and_click(driver, 'button', 'type', 'submit')

def input_user_pass_login(driver, username, password):
    username_elem = None
    password_elem = None
    input_elems = driver.find_elements_by_tag_name('input')
    for i in input_elems:
        if i.get_attribute('placeholder')=='Username':
            username_elem = i
        if i.get_attribute('placeholder')=='Password':
            password_elem = i
    if username_elem==None:
        print('Failed to find username field')
    if password_elem==None:
        print('Failed to find password field')
    if username_elem!=None and password_elem!=None:
        username_elem.send_keys(username)
        password_elem.send_keys(password)
    return username_elem, password_elem
    
def loginMoonBoard(driver, url="https://www.moonboard.com/", username='', password='', cookies_path=''):
    driver.get(url)
    loaded_cookies = False
    if os.path.exists(cookies_path):
        try:
            load_cookies(driver, cookies_path)
            loaded_cookies = True
        except:
            print('Cookies expired')
    
    login_elem = click_login_area(driver)
    if login_elem==None:
        return
    if loaded_cookies == True:
        relogin_elem = click_relogin(driver)
        if relogin_elem!=None:
            return
    
    username_elem, password_elem = input_user_pass_login(driver, username, password)
    login_button = click_login_button(driver)
    if username_elem==None or password_elem==None or login_button==None:
        return
    if cookies_path!='':
        cookies = driver.get_cookies()
        pickle.dump(cookies, open(cookies_path,'wb'))
    return

def check_fail_url(driver, url):
    driver.get(url)
    elems = find_element(driver, 'span', 'class', 'field-validation-error', num_tries=1)
    if elems!=None:
        return True
    return False