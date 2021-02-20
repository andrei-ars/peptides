# based on gridworld.py and tictactoe.py
"""
Requirements
pip install lxml
"""

import time
import datetime
import os
import sys
import logging
import numpy as np
import torch
# for Selenium_webdriver
from bs4 import BeautifulSoup
from xpath_soup import xpath_soup
from datagen import DataGenerator
from logger import Logger


class PageElement():
    def __init__(self, element_params=None, element_id=None, element_name=None, element_type=None, text=None, 
                    selenium_type=None, xpath=None):
        if element_params:
            self.params = element_params
            self.id = element_params['id']
            self.name = element_params['name']
            self.type = element_params['type']
            self.text = element_params['text'] 
            self.selenium_type = element_params['selenium_type']
            self.xpath = element_params['xpath']
        else:
            self.id = element_id,
            self.name = element_name,
            self.type = element_type,
            self.text = text,
            self.selenium_type = selenium_type,
            self.xpath = xpath


class SeleniumWebDriver():
    def __init__(self, init_url_address, driver_type="Chrome"):
        """ Return chrome webdriver and open the initial webpage.
        """
        print("WebDriver Initialization")
        self.delay_after_click = 3

        self.logger = Logger("_log.log")
        self.log = self.logger.log
        # self.log("Initialization")

        self.init_url_address = init_url_address
        from selenium import webdriver
        self.webdriver = webdriver
        self.options = self.webdriver.ChromeOptions()
        self.options.add_argument('--ignore-certificate-errors')
        self.options.add_argument("--test-type")
        self.options.add_argument("--silent")
        #options.add_argument("--headless")  # without openning a browser window
        #options.add_experimental_option("excludeSwitches", ["enable-automation"])
        #options.add_experimental_option('useAutomationExtension', False)
        #from webdriverwrapper import Chrome
        #self.driver = Chrome(options=options)
        # Open a website
        #window_before = self.driver.window_handles[0]
        self.driver = self.webdriver.Chrome(chrome_options=self.options)
        #self.driver = self.webdriver_simulation()
        self.data_generator = DataGenerator(self.driver)
        self.reset(initial=True)

    def reset(self, initial=False):
        """ This function doesn't reboot (reopen) browser each time but only if necessary.
        Set initial is True for the first time only.
        """
        self.log("RESET")
        #self.driver.get(self.init_url_address)
        self.history = []
        self.page_elements = None
        self.is_page_has_been_updated = False
        self.previos_html_code = None
        current_url = self._get_current_url()
        if not initial and current_url != "https://demo1.testgold.dev/login":
            self.log("self.driver.current_url={}".format(self.driver.current_url))
            self.log("REOPEN BROWSER")
            self.driver.close()
            self.driver = self.webdriver.Chrome(chrome_options=self.options)
            self.data_generator = DataGenerator(self.driver)
        self.log("self.init_url_address: {}".format(self.init_url_address))
        self.driver.get(self.init_url_address)

    
    def get_site_elements(self):
        #logging.debug("get_site_elements")
        html_code = self._get_html_code()
        elements = self._get_page_elements(self.html_code)
        self.site_elements = {
            'clickables': [elem.name for elem in elements['button']],
            'selectables': [], 
            'enterables': [elem.name for elem in elements['input']]
            }
        print("site_elements: {}".format(self.site_elements))
        return self.site_elements

    def action_on_element(self, action, element_number, data=None):
        """
        """
        print("action={}, element_number={}".format(action, element_number))
        action_to_element_type = {"CLICK": "button", "ENTER": "input"}
        element_type = action_to_element_type[action]
        element = self.page_elements[element_type][element_number]

        driver_element = self.driver.find_element_by_xpath(element.xpath)
        if action == "CLICK":
            print("Click on the element #{} with xpath={}".format(element_number, element.xpath))
            driver_element.click()
            self.logger.log("CLICK [{}]".format(element_number))
            time.sleep(self.delay_after_click)
        elif action == "ENTER":
            # driver_element.clear()
            data = self.data_generator.infer(element, self.html_code)
            print("Enter into the element #{} with xpath={}".format(element_number, element.xpath))
            driver_element.send_keys(data)
            self.logger.log("ENTER [{} ({})]: data=\"{}\"".format(element_number, element.name, data))
        return True

    """
    def click(self, current_element):
        #self.driver.find_element_by_xpath()
        element = self.driver.find_element_by_name(current_element)
        # the function find_element_by_name should be implemented
        element.click()

    def enter(self, current_element, data):
        element = self.driver.find_element_by_name(current_element)
        enter_field = self.driver.find_element_by_xpath("//input[@name='{}']".format(element))
        enter_field.clear()
        data = generate_data()
        enter_field.send_keys(data)
    """

    def is_target_achieved(self, targets=None):
        #return self.is_page_has_been_updated

        current_url = self._get_current_url()
        #print("current_url:", current_url)

        if targets:
            is_achieved = False
            final_targets = targets.get('final')
            for target in final_targets:
                target_url = target.get('url')
                if current_url == target_url:
                    print("current_url is target_url:", target_url)
                    self.logger.log("!current_url is target_url: {}".format(target_url))
                    is_achieved = True
        else:
            elements = self._get_page_elements(self.html_code)
            inputs = elements['input']
            singin_window = False
            for input0 in inputs:
                if input0.name == "password":
                    singin_window = True
            if not singin_window:
                print("The target is achieved. This is not a singin_window")
            is_achieved = not singin_window

        return is_achieved


    def _get_html_code(self):
        self.html_code = self.driver.page_source
        if self.previos_html_code is not None and self.previos_html_code != self.html_code:
            self.is_page_has_been_updated = True
            self.logger.log("page_has_been_updated")
        self.previos_html_code = self.html_code
        return self.html_code

    def _get_page_elements(self, html_code):
        html_soup = BeautifulSoup(html_code, 'lxml')
        elems = {}
        elems['input'] = html_soup.find_all('input')
        elems['button'] = html_soup.find_all('button')
        elements = {'input': [], 'button': []}

        for i, elem in enumerate(elems['input']):
            element_dict = {
                'id': i, # 0, 1, 2, ...
                'name': str(elem.attrs.get('name')), # like 'email'
                'type': str(elem.attrs.get('type')), # like 'email'
                'text': str(elem.attrs.get('placeholder')), # like 'email'
                'selenium_type': str(elem.name),  # like 'input'
                'xpath': xpath_soup(elem),
                }
            element = PageElement(element_params=element_dict)
            elements['input'].append(element)
            #print("input_element: {}".format(element))

        for i, elem in enumerate(elems['button']):
            element_dict = {
                'id': i,  # 0, 1, 2, ...
                'name': str(elem.attrs.get('class')), # like ['btn', 'btn-primary']
                'type': str(elem.attrs.get('type')), # like 'submit'
                'text': str(elem.attrs.get('placeholder')), # None
                'selenium_type': str(elem.name),  # like 'button'
                'xpath': xpath_soup(elem),
                }
            element = PageElement(element_params=element_dict)
            elements['button'].append(element)
            #print("button_element: {}".format(element))

            

        self.page_elements = elements
        return elements

    def _get_current_url(self):
        return self.driver.current_url


if __name__ == "__main__":

    driver = SeleniumWebDriver(init_url_address="https://demo1.testgold.dev/login")
    elements = driver.get_site_elements()
    print(elements)
    #driver.action_on_element()
    driver.action_on_element("ENTER", 0)
    driver.action_on_element("ENTER", 1)
    driver.action_on_element("CLICK", 0)
    
    while len(driver.get_site_elements()['clickables']) == 0:
        time.sleep(10)
    elements = driver.get_site_elements()
    print(elements)

    driver.action_on_element("CLICK", 0)
    