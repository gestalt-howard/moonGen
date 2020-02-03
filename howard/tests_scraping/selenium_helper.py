from selenium import webdriver
import pickle
import time

def load_driver(executable_path='/Users/aaronwu/Documents/misc/tmp/selenium/chromedriver'):
    driver = webdriver.Chrome(executable_path=executable_path)
    return driver

def load_cookies(driver, cookies_path):
    cookies = pickle.load(open(cookies_path,'rb'))
    for cookie in cookies:
        driver.add_cookie(cookie)
    return

def find_element_attr(driver, tag_name, attribute):
    values = None   
    try:
        elems = driver.find_elements_by_tag_name(tag_name)
        values = [e.get_attribute(attribute) for e in elems]
        if all(v==None for v in values):
            values=None
    except:
        pass
    if values==None:
        print('Failed to find ' + str(attribute))
    return values

def find_element_text(driver, tag_name):
    values = None   
    try:
        elems = driver.find_elements_by_tag_name(tag_name)
        values = [e.text for e in elems]
        if all(v==None for v in values):
            values=None
    except:
        pass
    if values==None:
        print('Failed to find ' + str(tag_name))
    return values

def find_element(driver, tag_name, attribute, value, num_tries = 10):
    elem = None   
    for i in range(num_tries):
        try:
            elems = driver.find_elements_by_tag_name(tag_name)
            for e in elems:
                if e.get_attribute(attribute)==value:
                    elem=e
                    break
            if elem!=None:
                break
        except:
            time.sleep(.5)
            continue
    if elem==None:
        print('Failed to find ' + str(attribute) + ' ' + str(value))
    return elem

def find_and_click(driver, tag_name, attribute, value, text=False, num_tries = 10):
    elem = None
    for i in range(num_tries):
        try:
            if text:
                elem = find_text(driver, tag_name, value)
            else:
                elem = find_element(driver, tag_name, attribute, value)
            if elem!=None:
                elem.click()
                break
        except:
            time.sleep(.5)
            continue
    if elem==None:
        print('Failed to click')
    return elem

def find_text(driver, tag_name, text, num_tries = 10):
    elem = None   
    for i in range(num_tries):
        try:
            elems = driver.find_elements_by_tag_name(tag_name)
            for e in elems:
                if e.text==text:
                    elem=e
                    break
            if elem!=None:
                break
        except:
            time.sleep(.5)
            continue
    if elem==None:
        print('Failed to find ' + str(text))
    return elem

def get_elem_set(driver, tag_name, attr_dict, num_tries = 10):
    elem_set = []
    for i in range(num_tries):
        try:
            elems = driver.find_elements_by_tag_name(tag_name)
            for e in elems:
                if all(e.get_attribute(attr)==attr_dict[attr] for attr in attr_dict):
                    elem_set.append(e)
            if len(elem_set)>0:
                break
        except:
            time.sleep(.5)
            continue
    return elem_set