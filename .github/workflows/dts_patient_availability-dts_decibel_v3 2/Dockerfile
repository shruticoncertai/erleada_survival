FROM eurekahealth.jfrog.io/concertai/pythonbaseimages/python3.10_baseimage:v1
RUN yum update -y
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

#copying requirements.txt file
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy over code base
COPY ./src /app/src
COPY ./config /app/config
COPY config/entrypoint.sh /entrypoint.sh
COPY config/nginx_dpe.conf /etc/nginx/conf.d/
COPY certs/ /etc/ssl/
RUN rm -f /etc/nginx/http.d/default.conf
WORKDIR /app/src

RUN chmod +x /entrypoint.sh
EXPOSE 6019 443
ENTRYPOINT ["sh" , "/entrypoint.sh"]